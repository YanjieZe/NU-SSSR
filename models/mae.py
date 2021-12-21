import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules.linear import Identity

from models.vit import *

from PIL import Image
import os

class MAE(nn.Module):
    def __init__(
        self, encoder: ViT, decoder_dim,
        mask_ratio=0.75, decoder_depth=1,
        num_decoder_heads=8, decoder_dim_per_head=64
    ):
        super().__init__()
        assert 0. < mask_ratio < 1., f'mask ratio should be between 0 and 1, got: {mask_ratio}'
        
        self.encoder = encoder
        self.patch_h, self.patch_w = encoder.patch_h, encoder.patch_w
        
        num_patches_plus_cls_token, encoder_dim = encoder.pos_embed.shape[-2:]
        
        num_pixels_per_patch = encoder.patch_embed.weight.size(1)
        
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        
        self.mask_ratio = mask_ratio
        
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))
        
        self.decoder = Transformer(
            decoder_dim,
            decoder_dim * 4,
            depth=decoder_depth,
            num_heads=num_decoder_heads,
            dim_per_head=decoder_dim_per_head
        )
        
        self.decoder_pos_embed = nn.Embedding(num_patches_plus_cls_token - 1, decoder_dim)
        
        self.head = nn.Linear(decoder_dim, num_pixels_per_patch)
        
    def forward(self, x: torch.Tensor):
        device = x.device
        b, c, h, w = x.shape
        
        num_patches = (h // self.patch_h) * (w // self.patch_w)
        # print(x.shape)
        # print(h, w, self.patch_h, self.patch_w)
        patches = x.view(
            b, c,
            h // self.patch_h, self.patch_h,
            w // self.patch_w, self.patch_w
        )
        # print(patches.shape)
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        # print(patches.shape)
        patches = patches.reshape(b, num_patches, -1)
        #* (b, c, h, w) -> (b, c, N_h, h_p, N_w, w_p) ->
        #* (b, N_h, N_w, h_p, w_p, c) -> (b, N_h, N_w, h_p*w_p*c)
        
        num_masked = int(self.mask_ratio * num_patches)
        
        shuffled_indices = torch.rand(b, num_patches, device=device).argsort()
        mask_ind, unmask_ind = shuffled_indices[:, :num_masked], shuffled_indices[:, num_masked:]
        batch_ind = torch.arange(b, device=device).unsqueeze(1)
        mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]
        
        unmask_tokens = self.encoder.patch_embed(unmask_patches)
        unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind + 1]
        
        encoded_tokens = self.encoder.transformer(unmask_tokens)
        
        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)
        
        mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
        mask_tokens += self.decoder_pos_embed(mask_ind)
        
        concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
        dec_input_tokens = torch.empty_like(concat_tokens, device=device)
        dec_input_tokens[batch_ind, shuffled_indices] = concat_tokens
        
        decoded_tokens = self.decoder(dec_input_tokens)
        
        dec_mask_tokens = decoded_tokens[batch_ind, mask_ind, :]
        pred_mask_pixel_values = self.head(dec_mask_tokens)
        
        loss = torch.nn.functional.mse_loss(pred_mask_pixel_values, mask_patches)
        
        return loss
    
    def forward_nomask(self, x: torch.Tensor):
        device = x.device
        b, c, h, w = x.shape
        
        num_patches = (h // self.patch_h) * (w // self.patch_w)
        
        patches = x.view(
            b, c,
            h // self.patch_h, self.patch_h,
            w // self.patch_w, self.patch_w
        ) \
        .permute(0, 2, 4, 3, 5, 1) \
        .reshape(b, num_patches, -1)
        #* (b, c, h, w) -> (b, c, N_h, h_p, N_w, w_p) ->
        #* (b, N_h, N_w, h_p, w_p, c) -> (b, N_h, N_w, h_p*w_p*c)
        
        num_masked = 0
        
        shuffled_indices = torch.rand(b, num_patches, device=device).argsort()
        unmask_ind = shuffled_indices
        batch_ind = torch.arange(b, device=device).unsqueeze(1)
        unmask_patches = patches[batch_ind, shuffled_indices]
        
        unmask_tokens = self.encoder.patch_embed(unmask_patches)
        unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind + 1]
        
        encoded_tokens = self.encoder.transformer(unmask_tokens)
        
        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)
        
        mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
        
        concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
        dec_input_tokens = torch.empty_like(concat_tokens, device=device)
        dec_input_tokens[batch_ind, shuffled_indices] = concat_tokens
        
        decoded_tokens = self.decoder(dec_input_tokens)
        
        dec_mask_tokens = decoded_tokens
        pred_mask_pixel_values = self.head(dec_mask_tokens)
        
        # mse_per_patch = (pred_mask_pixel_values - mask_patches).abs().mean(dim=-1)
        # mse_all_patches = mse_per_patch.mean()
        
        recons_patches = patches.detach()
        recons_patches = pred_mask_pixel_values
        recons_img = recons_patches.view(
            b, h // self.patch_h, w // self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        return recons_img
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        self.eval()
        
        device = x.device
        b, c, h, w = x.shape
        
        num_patches = (h // self.patch_h) * (w // self.patch_w)
        
        patches = x.view(
            b, c,
            h // self.patch_h, self.patch_h,
            w // self.patch_w, self.patch_w
        ) \
        .permute(0, 2, 4, 3, 5, 1) \
        .reshape(b, num_patches, -1)
        #* (b, c, h, w) -> (b, c, N_h, h_p, N_w, w_p) ->
        #* (b, N_h, N_w, h_p, w_p, c) -> (b, N_h, N_w, h_p*w_p*c)
        
        num_masked = int(self.mask_ratio * num_patches)
        
        shuffled_indices = torch.rand(b, num_patches, device=device).argsort()
        mask_ind, unmask_ind = shuffled_indices[:, :num_masked], shuffled_indices[:, num_masked:]
        batch_ind = torch.arange(b, device=device).unsqueeze(1)
        mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]
        
        unmask_tokens = self.encoder.patch_embed(unmask_patches)
        unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind + 1]
        
        encoded_tokens = self.encoder.transformer(unmask_tokens)
        
        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)
        
        mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
        mask_tokens += self.decoder_pos_embed(mask_ind)
        
        concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
        dec_input_tokens = torch.empty_like(concat_tokens, device=device)
        dec_input_tokens[batch_ind, shuffled_indices] = concat_tokens
        
        decoded_tokens = self.decoder(dec_input_tokens)
        
        dec_mask_tokens = decoded_tokens[batch_ind, mask_ind, :]
        pred_mask_pixel_values = self.head(dec_mask_tokens)
        
        # mse_per_patch = (pred_mask_pixel_values - mask_patches).abs().mean(dim=-1)
        # mse_all_patches = mse_per_patch.mean()
        
        recons_patches = patches.detach()
        recons_patches[batch_ind, mask_ind] = pred_mask_pixel_values
        recons_img = recons_patches.view(
            b, h // self.patch_h, w // self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)
        
        mask_patches = torch.zeros_like(mask_patches, device=mask_patches.device)
        
        patches[batch_ind, mask_ind] = mask_patches
        patches_to_img = patches.view(
            b, h // self.patch_h, w // self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)
        
        return recons_img, patches_to_img
