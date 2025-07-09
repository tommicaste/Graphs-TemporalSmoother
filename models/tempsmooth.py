from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import aggr
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv,SAGEConv,GraphConv,TransformerConv,ChebConv,GATConv,SGConv,GeneralConv
from torch.nn import Conv1d, MaxPool1d, ModuleList
import random
import math
from tqdm import tqdm
import torch.optim as optim


class SpatialConvLayer(nn.Module):
    """
    Spatial convolution layer to take multiple convolution options
    """
    def __init__(self, in_channels, out_channels, conv_operator, activation=nn.ReLU()):
        super(SpatialConvLayer, self).__init__()
        self.conv = conv_operator(in_channels, out_channels)
        self.activation = activation

    def forward(self, x, edge_index):
        
        x = self.conv(x, edge_index)

        if self.activation is not None:
            x = self.activation(x)
        return x
    

class TemporalTransformer(nn.Module):
    """
    Temporal Transformer Function to merge embeddings
    """
    def __init__(self, embed_dim, num_heads=1, num_layers=1):
        super(TemporalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, H_tilde, H_prev):

        
        x = torch.stack([H_prev, H_tilde], dim=1)
        
        
        out = self.transformer(x)
        
        H_out = out.mean(dim=1)
        return H_out



class DynamicSpatialTemporalClassifier(nn.Module):
    def __init__(
        self,
        in_channels,        
        embedding_sizes,     
        mlp_sizes,           
        num_classes,         
        conv_operator=GCNConv,
        attn_heads=None,     
        activation=nn.ReLU,
        pooling='mean'       
    ):
        super(DynamicSpatialTemporalClassifier, self).__init__()
        
        
        if attn_heads is None:
            attn_heads = [1] * len(embedding_sizes)
        if len(attn_heads) < len(embedding_sizes):
            attn_heads += [1] * (len(embedding_sizes) - len(attn_heads))
        
        self.num_layers = len(embedding_sizes)
        self.layers_conv = nn.ModuleList()
        self.layers_attn = nn.ModuleList()
        current_in = in_channels
        

        for i, out_dim in enumerate(embedding_sizes):
            self.layers_conv.append(
                SpatialConvLayer(
                    in_channels=current_in,
                    out_channels=out_dim,
                    conv_operator=conv_operator,
                    activation=activation()
                )
            )
            self.layers_attn.append(
                TemporalTransformer(embed_dim=out_dim, num_heads=attn_heads[i])
            )
            current_in = out_dim
        
        
        self.pooling = pooling
        mlp_layers = []
        prev_dim = embedding_sizes[-1]  
        for hidden_dim in mlp_sizes:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(activation())
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, snapshots):
        
        prev_embeddings = [None] * self.num_layers
        
        
        for t, data in enumerate(snapshots):
            x = data.x
            edge_index = data.edge_index
            
            for i in range(self.num_layers):
                
                tilde = self.layers_conv[i](x, edge_index)
                
                if t == 0:
                    h = tilde
                else:
                    h = self.layers_attn[i](tilde, prev_embeddings[i])
                x = h
                prev_embeddings[i] = h
        
        if self.pooling == 'mean':
            pooled = x.mean(dim=0) 
        elif self.pooling == 'max':
            pooled, _ = x.max(dim=0) 
        pooled = pooled.unsqueeze(0)  
        h_mlp = self.mlp(pooled)      
        logits = self.output_layer(h_mlp)
        return logits
