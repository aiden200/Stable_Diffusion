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
    
    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

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
    
    def _get_prev_timestep(self, timestep: int) -> int:
        # Getting prev timestep given the step
        return timestep - (self.num_training_steps // self.num_inference_steps)
    
    def _get_var(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_prev_timestep(timestep)
        alpha_prod_t = self.alpha_cum_prod[timestep]
        alpha_prod_t_prev = self.alpha_cum_prod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        variance = (1- alpha_prod_t_prev) / (1-alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-25)
        return variance

    def step(self, timestep: int, x_t: torch.Tensor, model_output: torch.Tensor):
        # x_t | x_t-1
        t = timestep
        prev_t = self._get_prev_timestep(t)
        
        alpha_prod_t = self.alpha_cum_prod[timestep]
        alpha_prod_t_prev = self.alpha_cum_prod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1-alpha_prod_t_prev
        curr_alpha_t = alpha_prod_t / alpha_prod_t_prev
        curr_beta_t = 1- curr_alpha_t

        pred_x_0 = (x_t - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # To predict the noise, we have to estimate the gaussian noise to remove. Predict mean & variance

        # now we compute the coefficients of x_t and x_0
        pred_x_0_coeff = (alpha_prod_t_prev ** 0.5 * curr_beta_t) / (beta_prod_t) 
        pred_x_t_coeff = curr_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # Predicted previous sample mean
        pred_prev_mean = pred_x_0_coeff * pred_x_0 + pred_x_t_coeff * x_t


        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator = self.generator, device=device, dtype=model_output.dtype)
            var = (self._get_var(t) ** .5) * noise
        
        # Transforming N(0,1) -> N(mu, sigma**2)
        # x = mu + sigma * Z where Z ~ N(0,1)

        predicted_previous_sample = pred_prev_mean + var
        return predicted_previous_sample
            

