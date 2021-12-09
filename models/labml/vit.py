import torch
from torch import nn

class PatchEmbeddings(nn.Module):
    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=d_model, 
                              kernel_size=patch_size, stride=patch_size)
    
    #* x.shape = [B, C, H, W]
    def forward(self, x: torch.Tensor):
        x = self.conv(x)             #* x.shape = [B, C, H/P, W/P]
        b, c, h, w = x.shape
        x = x.permute(2, 3, 0, 1)    #* x.shape = [H/P, W/P, B, C]
        x = x.reshape(h*w, b, c)     #* x.shape = [H/P * W/P, B, C]
        return x
    
    
class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5_000):
        super().__init__()
        self.positional_embeddings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)
        
    #* x.shape =[H/P * W/P, B, C]
    def forward(self, x: torch.Tensor):
        pe = self.positional_embeddings[x.shape[0]]
        return x + pe


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, n_hidden: int, n_classes: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, n_hidden)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_classes)
        
    def forward(self, x: torch.Tensor):
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, transformer_layer: int, n_layers: int, patch_emb: PatchEmbeddings, pos_emb: LearnedPositionalEmbeddings, classification: ClassificationHead):
        super().__init__()
        self.patch_emb = patch_emb
        self.pos_emb = pos_emb
        self.classification = classification
        
        self.transformer_layers = transformer_layer
        
        
        