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
        
