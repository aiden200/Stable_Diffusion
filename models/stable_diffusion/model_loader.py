from models.clip.clip_decoder import CLIP
from models.vae.encoder import VAE_encoder
from models.vae.decoder import VAE_Decoder
from models.unet.diffusion import Diffusion
from models.stable_diffusion import model_converter

def preload_models_from_weights(path, device):
    # Need to convert the names of the model to the pretrained names
    state_dict = model_converter.load_from_standard_weights(path, device)

    encoder = VAE_encoder().to(device)
    encoder.load_state_dict(state_dict["encoder"], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion
    }