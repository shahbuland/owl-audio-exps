import sys
import os

import torch
from diffusers import AutoencoderDC

sys.path.append("./owl-vaes")
from owl_vaes import from_pretrained

def _get_decoder_only():
    model = load_proxy_model(
        "../checkpoints/128x_proxy_titok.yml",
        "../checkpoints/128x_proxy_titok.pt",
        "../checkpoints/16x_dcae.yml",
        "../checkpoints/16x_dcae.pt"
    )
    del model.transformer.encoder
    return model

def get_decoder_only(vae_id, cfg_path, ckpt_path):
        if vae_id == "dcae":
            model_id = "mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers"
            model = AutoencoderDC.from_pretrained(model_id).bfloat16().cuda().eval()
            del model.encoder
            return model.decoder
        else:
            model = from_pretrained(cfg_path, ckpt_path)
            del model.encoder
            model = model.decoder
            model = model.bfloat16().cuda().eval()
            return model

@torch.no_grad()
def make_batched_decode_fn(decoder, batch_size = 8):
    def decode(x):
        # x is [b,n,c,h,w]
        b,n,c,h,w = x.shape
        x = x.view(b*n,c,h,w).contiguous()

        batches = x.split(batch_size)
        batch_out = []
        for batch in batches:
            batch_out.append(decoder(batch).bfloat16())

        x = torch.cat(batch_out) # [b*n,c,h,w]
        _,c,h,w = x.shape
        x = x.view(b,n,c,h,w).contiguous()

        return x
    return decode

@torch.no_grad()
def make_batched_audio_decode_fn(decoder, batch_size=8, max_seq_len=120):
    def decode(x):
        # x is [b,n,c] audio latents
        b, n, c = x.shape
        
        # If sequence is within VAE's training length, use original approach
        if n <= max_seq_len:
            x = x.transpose(1, 2)  # [b,c,n]
            batches = x.contiguous().split(batch_size)
            batch_out = []
            for batch in batches:
                batch_out.append(decoder(batch).bfloat16())
            
            x = torch.cat(batch_out)  # [b,c,n]
            x = x.transpose(-1, -2).contiguous()  # [b,n,2]
            return x
        
        # For longer sequences, use sliding window approach
        x = x.transpose(1, 2)  # [b,c,n]
        
        # Split into windows of max_seq_len
        windows = []
        for start in range(0, n, max_seq_len):
            end = min(start + max_seq_len, n)
            window = x[:, :, start:end]  # [b,c,window_len]
            
            # Batch this window across batch dimension
            batches = window.contiguous().split(batch_size)
            batch_out = []
            for batch in batches:
                batch_out.append(decoder(batch).bfloat16())
            
            window_decoded = torch.cat(batch_out)  # [b,c,decoded_len]
            windows.append(window_decoded)
        
        # Concatenate all windows along sequence dimension
        x = torch.cat(windows, dim=2)  # [b,c,total_decoded_len]
        x = x.transpose(-1, -2).contiguous()  # [b,total_decoded_len,2]
        
        return x
    return decode

def get_audio_encoder_decoder(cfg_path, ckpt_path):
    """Get both encoder and decoder for audio VAE using from_pretrained"""
    model = from_pretrained(cfg_path, ckpt_path)
    
    encoder = model.encoder.bfloat16().cuda().eval()
    decoder = model.decoder.bfloat16().cuda().eval()
    return encoder, decoder

@torch.no_grad()
def make_batched_audio_encode_fn(encoder, batch_size=8):
    """Create batched encoding function for audio waveforms"""
    def encode(x):
        # x is [b, n_samples, 2] raw waveforms
        x = x.transpose(1, 2)  # [b, 2, n_samples]
        b, c, n = x.shape
        
        batches = x.contiguous().split(batch_size)
        batch_out = []
        for batch in batches:
            batch_out.append(encoder(batch).bfloat16())
        
        x = torch.cat(batch_out)  # [b, latent_channels, n_latents] 
        x = x.transpose(-1, -2).contiguous()  # [b, n_latents, latent_channels]
        
        return x
    return encode