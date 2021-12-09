import math
import torch
import torch.nn as nn

def get_positional_encoding(d_model: int, max_len: int = 5000):
    encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    
    return encodings

class EmbeddingsWithPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len))
        
    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].requires_grad_(False)
        return self.linear(x) + math.sqrt(self.d_model) + pe

class EmbeddingsLearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.positional_embeddings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)
        
    def forward(self, x: torch.Tensor):
        pe = self.positional_embeddings[:x.shape[0]]
        return self.linear(x) * math.sqrt(self.d_model) + pe

# class TransformerLayer(nn.Module):
#     def __init__(self, *,
#                  d_model: int,
#                  self_attn: )