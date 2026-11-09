# Denoising-Diffusion-Probabilistic-Model-for-generating-images-of-moon

This is a basic diffusion model that follows the DDPM architecture. I only used around 1000 grayscale images to train it into a 4 million parameter model.


I am not going to describe it all here again since almost everything is explained in the notebook (notebook.ipynb) and if you want to take a peek at the images go to /moon folder.

### Devlog:-

This part is not technical, I will tell my story of how I built it and tested it to improve forward.

I had started it off to understand generative AI and architectures like diffusion better. My original inspo was from a youtube video that was about generating images of cars only instead. I took it to even lower scale with 2 channel (grayscale) and single object. My first model was an abomination, I had forgot to clamp the time-steps and the noising steps kept going downstream. Here's the 64 sample batch from that first model:-

<img width="949" height="778" alt="Screenshot 2026-08-17 230627" src="https://github.com/user-attachments/assets/0d960112-f1af-4dc7-af4a-329273d608e5" />

After fixing it and training it from zero again, I got pretty good results:- 

<img width="943" height="795" alt="Screenshot 2026-08-18 133105" src="https://github.com/user-attachments/assets/9979213c-39ef-4e37-97db-8a381b8efbeb" />

I wanted to improve it further but at that point my data was the clear bottleneck and compute was obvious problem as well. I had trained it 100% on cloud for free in kaggle.com weekly 30hr GPU (Tesla T4 16gb) runtime. 
