import torch
import numpy as np
from tqdm import tqdm
from models.stable_diffusion.ddpm import DDPMSampler

W = 512
H = 512
ZW = W//8
ZH = H//8

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp: # cuts off anything out of range
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Doing positional encoding in time embedding
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None] 
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def generate(prompt: str, 
            unconditional_prompt: str, 
            input_image=None, 
            strength=0.8, # how much attention to pay to the input image or creativity
            cfg=True, # The model paying attention to the conditioned prompt
            cfg_scale=7.5, 
            sampler_name="ddpm", 
            n_inference_steps=50,
            models={}, 
            seed=None,
            device=None,
            idle_device=None,
            tokenizer=None):
    with torch.no_grad(): #inference
        if not (0<strength <=1):
            raise ValueError("Strength must be between 0 and 1")
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x
    
    generator = torch.Generator(device=device) # generating noise
    if seed is None:
        generate.seed()
    else:
        generator.manual_seed(seed)
    
    clip = models["clip"]
    clip.to(device)

    # Prompt -> tokens. pad to max length
    conditional_tokens = tokenizer.batch_encode([prompt], padding="max_length", max_length=77).input_ids
    #(B, Seq_length)
    conditional_tokens = torch.tensor(conditional_tokens, dtype=torch.long, device=device)
    conditional_context = clip(conditional_tokens) # (B, Seq_length, E)
    
    if cfg:
        #random noise
        unconditional_tokens = tokenizer.batch_encode([unconditional_prompt], padding="max_length", max_length=77).input_ids
        unconditional_tokens = torch.tensor(unconditional_tokens, dtype=torch.long, device=device)
        unconditional_context = clip(unconditional_tokens)

        #( 2, seq_length, E) (2, 77, 768)
        context = torch.cat([conditional_context, unconditional_context])
    else:
        context = conditional_context

    to_idle(clip) # offload GPU

    if sampler_name =="ddpm":
        sampler = DDPMSampler(generator)
        sampler.set_inference_steps(n_inference_steps) # how many states
    else:
        raise ValueError("Unknown Sampler")
    
    latents_shape = (1, 4, ZH, ZW)

    if input_image: # not random noise, image 2 image
        encoder = models["encoder"]
        encoder.to(device)
        input_image_tensor = input_image.resize((W, H))
        input_image_tensor = np.array(input_image_tensor)

        input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
        input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
        input_image_tensor = input_image_tensor.unsqueeze(0) # add batch dim
        input_image_tensor = input_image_tensor.permute(0,3,1,2)

        encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
        latents = encoder(input_image_tensor, encoder_noise)

        sampler.set_strength(strength=strength)
        latents = sampler.add_noise(latents, sampler.timesteps[0])
        to_idle(encoder)
    else: # text to image
        # Start with random Noise N(0, I)
        latents = torch.randn(latents_shape, generator=generator, device=device)
    
    diffusion = models["device"]
    diffusion.to(device)
    timesteps = tqdm(sampler.timesteps)
    for i, timestep in enumerate(timesteps):
        # ( 1, 320)
        time_embedding = get_time_embedding(timestep).to(device)
        # VAE encoder output is input (B, 4, ZH, ZW)
        model_input = latents

        if cfg:
            # (2*B, 4, ZH, ZW) add unconditional prompt too
            model_input = model_input.repeat(2,1,1,1)
        
        # Noise predicted by the model 
        model_output = diffusion(model_input, context, time_embedding)

        if cfg:
            output_conditioned, output_unconditioned = model_output.chunk(2) # divide by 2 to get both conditional and unconditional
            # w * (o_c - o_u) + o_u
            model_output = cfg_scale * (output_conditioned - output_unconditioned) + output_unconditioned

        # Remove the noise predicted by the UNET
        latents = sampler.step(timestep, latents, model_output)
    
    to_idle(diffusion)
    decoder = models["decoder"]
    decoder.to(device)

    images = decoder(latents)
    to_idle(decoder)

    images = rescale(images,(-1,1), (0,255), clamps=True)
    # (B, C, H, W) -> (B, H, W, C)
    images = images.permute(0, 2, 3, 1)
    images = images.to("cpu", torch.uint8).numpy()
    return images[0]