
import argparse, math, os
import numpy as np
import torch
from torch import nn
from PIL import Image

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        e = math.log(10000) / (half - 1)
        e = torch.exp(torch.arange(half, device=time.device) * -e)
        e = time[:, None] * e[None, :]
        return torch.cat((e.sin(), e.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, groups=8):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
            self.res_proj = nn.Conv2d(2 * in_ch, out_ch, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            self.res_proj = nn.Conv2d(in_ch, out_ch, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, t):
        residual = self.res_proj(x)
        h = self.norm1(self.act(self.conv1(x)))
        h = h + self.act(self.time_mlp(t))[..., None, None]
        h = self.norm2(self.act(self.conv2(h)))
        return self.transform(h + residual)


class SelfAttention(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax((q.transpose(1, 2) @ k) * self.scale, dim=-1)
        out = (v @ attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(out)


class MoonUnet(nn.Module):
    def __init__(self, image_channels=1, time_emb_dim=64):
        super().__init__()
        down, up = (32, 64, 128, 256), (256, 128, 64, 32)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU())
        self.conv0 = nn.Conv2d(image_channels, down[0], 3, padding=1)
        self.downs = nn.ModuleList(Block(down[i], down[i + 1], time_emb_dim) for i in range(3))
        self.bottleneck_attn = SelfAttention(down[-1])
        self.ups = nn.ModuleList(Block(up[i], up[i + 1], time_emb_dim, up=True) for i in range(3))
        self.output = nn.Conv2d(up[-1], image_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        skips = []
        for d in self.downs:
            x = d(x, t)
            skips.append(x)
        x = self.bottleneck_attn(x)
        for u in self.ups:
            x = u(torch.cat((x, skips.pop()), dim=1), t)
        return self.output(x)


def cosine_beta_schedule(timesteps, s=0.008):
    x = torch.linspace(0, timesteps, timesteps + 1)
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    return torch.clip(1 - (ac[1:] / ac[:-1]), 0.0001, 0.9999)


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


class Sampler:
    def __init__(self, model, t_steps, schedule, img_size, channels, device):
        self.model, self.T, self.size, self.ch, self.device = model, t_steps, img_size, channels, device
        betas = cosine_beta_schedule(t_steps) if schedule == "cosine" else linear_beta_schedule(t_steps)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, 0)
        ac_prev = torch.nn.functional.pad(ac[:-1], (1, 0), value=1.0)
        self.betas = betas.to(device)
        self.sqrt_ac = ac.sqrt().to(device)
        self.sqrt_1m_ac = (1 - ac).sqrt().to(device)
        self.sqrt_recip_alphas = (1.0 / alphas).sqrt().to(device)
        self.post_var = (betas * (1 - ac_prev) / (1 - ac)).to(device)

    @torch.no_grad()
    def step(self, x, i):
        t = torch.full((x.shape[0],), i, device=self.device, dtype=torch.long)
        eps = self.model(x, t)
        x0 = ((x - self.sqrt_1m_ac[i] * eps) / self.sqrt_ac[i]).clamp(-1, 1)
        eps = (x - self.sqrt_ac[i] * x0) / self.sqrt_1m_ac[i]
        mean = self.sqrt_recip_alphas[i] * (x - self.betas[i] * eps / self.sqrt_1m_ac[i])
        if i == 0:
            return mean
        return mean + self.post_var[i].sqrt() * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, n, generator=None):
        x = torch.randn((n, self.ch, self.size, self.size), device=self.device, generator=generator)
        for i in reversed(range(self.T)):
            x = self.step(x, i)
        return x


def to_uint8(x):
    x = ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte().cpu().numpy()  # N,C,H,W
    return x[:, 0] if x.shape[1] == 1 else x.transpose(0, 2, 3, 1)


def main():
    ap = argparse.ArgumentParser(description="Sample moon images from a DDPM .pt checkpoint")
    ap.add_argument("--ckpt", default="best_checkpoint.pt")
    ap.add_argument("--n", type=int, default=16, help="number of images")
    ap.add_argument("--out", default="samples", help="output directory")
    ap.add_argument("--batch", type=int, default=64, help="images per forward pass (lower if out of memory)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--grid", action="store_true", help="also save one grid.png")
    ap.add_argument("--upscale", type=int, default=1, help="nearest-neighbour upscale factor for saved PNGs")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    ck = torch.load(a.ckpt, map_location="cpu")
    if isinstance(ck, dict) and "model" in ck and "config" in ck:   # new self-contained format
        cfg, state = ck["config"], ck["model"]
    else:                                                          # raw state_dict from the original notebook
        cfg = dict(img_size=64, channels=1, time_emb_dim=64, t_steps=150, schedule="cosine")
        state = ck
    model = MoonUnet(cfg["channels"], cfg["time_emb_dim"]).to(device)
    model.load_state_dict(state)
    model.eval()
    sampler = Sampler(model, cfg["t_steps"], cfg["schedule"], cfg["img_size"], cfg["channels"], device)

    gen = None
    if a.seed is not None:
        gen = torch.Generator(device=device).manual_seed(a.seed)
        torch.manual_seed(a.seed)

    os.makedirs(a.out, exist_ok=True)
    imgs, done = [], 0
    while done < a.n:
        b = min(a.batch, a.n - done)
        print(f"sampling {done + 1}-{done + b} of {a.n} on {device} ...")
        imgs.append(to_uint8(sampler.sample(b, gen)))
        done += b
    imgs = np.concatenate(imgs)

    for k, im in enumerate(imgs):
        p = Image.fromarray(im)
        if a.upscale > 1:
            p = p.resize((p.width * a.upscale,) * 2, Image.NEAREST)
        p.save(os.path.join(a.out, f"moon_{k:04d}.png"))

    if a.grid:
        cols = int(math.ceil(math.sqrt(len(imgs))))
        rows = int(math.ceil(len(imgs) / cols))
        s = imgs.shape[1]
        g = np.zeros((rows * s, cols * s), dtype=np.uint8)
        for k, im in enumerate(imgs):
            r, c = divmod(k, cols)
            g[r * s:(r + 1) * s, c * s:(c + 1) * s] = im
        Image.fromarray(g).save(os.path.join(a.out, "grid.png"))
    print(f"saved {len(imgs)} image(s) to {a.out}/")


if __name__ == "__main__":
    main()
