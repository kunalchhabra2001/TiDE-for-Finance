import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import DistilBertModel
import pandas as pd
from transformers import DistilBertTokenizer
from tqdm import tqdm
import statistics
import numpy as np
import random

class MLPResidual(torch.nn.Module):
  """Simple one hidden state residual network."""

  def __init__(
      self, input_dim, hidden_dim, output_dim, device, layer_norm=False, dropout_rate=0.0,
  ):
    super(MLPResidual, self).__init__()
    self.lin_a = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim, device=device),
        torch.nn.ReLU()
    )
    self.lin_b = torch.nn.Linear(hidden_dim, output_dim, device=device)
    self.lin_res = torch.nn.Linear(input_dim, output_dim, device=device)
    if layer_norm:
      self.lnorm = torch.nn.LayerNorm() # which dimension here?
    self.layer_norm = layer_norm
    self.dropout = torch.nn.Dropout(dropout_rate)

  def forward(self, inputs):
    """Call method."""
    h_state = self.lin_a(inputs)
    out = self.lin_b(h_state)
    out = self.dropout(out)
    res = self.lin_res(inputs)
    if self.layer_norm:
      return self.lnorm(out + res)
    return out + res

def make_dnn_residual(input_dim, hidden_dims, device, layer_norm=False, dropout_rate=0.0):
  """Multi-layer DNN residual model."""
  if len(hidden_dims) < 2:
    return torch.nn.Linear(
        input_dim,
        hidden_dims[-1],
        device=device,
    )
  layers = []
  for i, hdim in enumerate(hidden_dims[:-1]):
    if i==0:
      prev_hidden_dim = input_dim
    else:
      prev_hidden_dim =  hidden_dims[i - 1]
    layers.append(
        MLPResidual(
            prev_hidden_dim,
            hdim,
            hidden_dims[i + 1],
            device=device,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
        )
    )
  return torch.nn.Sequential(*layers)

class TextModel(torch.nn.Module):
    def __init__(self, device, text_dim=64):
        super(TextModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(768, 256, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(256, text_dim, device=device)
        )

    def forward(self, text):
        outputs = self.bert(text['input_ids'], text['attention_mask'])
        embedded_text = self.linear(outputs['last_hidden_state'][:,0,:])
        return embedded_text

class LocationModel(torch.nn.Module):
    def __init__(self, device, location_dim=32, latlng_feat_size=19):
        super(LocationModel, self).__init__()
        self.addr_proj = torch.nn.Linear(768, 128)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(128 + latlng_feat_size, 64, device=device), # lat-lng feature size 19 (in latlng_feat_size)
            torch.nn.ReLU(),
            torch.nn.Linear(64, location_dim, device=device),
            torch.nn.ReLU()
        )

    def forward(self, loc_bert_emb, bounds):
        geo_features = torch.cat([self.addr_proj(loc_bert_emb), bounds], dim=2)
        return self.linear(geo_features.type(torch.float32))
