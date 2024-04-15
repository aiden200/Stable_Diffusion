import torch
import numpy as np


class DDPMSampler:
    def __init__(self, 
                 generator: torch.Generator, 
                 num_training_steps = 1000, 
                 beta_start: float=0.00085, # how the gaussian distribution in each state of the MDP varies when adding noise at each step 
                 beta_end: float = 0.012 # Choices by authors
                 ):
        self.betas = torch.linspace(beta_start**.5, beta_end**.5, num_training_steps, dtype=torch.float32) ** 2 # in paper
        self.alphas = 1.0 - self.betas # Alpha that uses for closed form solution in timestep noise
        self.alpha_cum_prod = torch.cumprod(self.alphas, 0)
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
    
    def set_inference_timesteps(self, num_inference_steps=50):
        # num_inference_steps is the step size to num_training_steps (1000)
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cum_prod = self.alpha_cum_prod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        # The actual math in the forward pass (closed formula for adding noise)
        sqrt_alpha_prod = alpha_cum_prod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = (1 - alpha_cum_prod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod) * noise
        return noisy_samples
