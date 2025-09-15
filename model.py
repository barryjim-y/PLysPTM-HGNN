#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from logzero import logger


class MultiFeatureGCN(nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super(MultiFeatureGCN, self).__init__()


        self.hidden_dim = hidden_dim

        logger.info(f"Initializing MultiFeatureGCN (ablation experiment - removing feature transformation layer), feature dimensions: {input_dims}, "
                    f"hidden_dim={hidden_dim}, output_dim={output_dim}, n_layers={n_layers}")


        self.ipr_transform = nn.Linear(input_dims['ipr'], hidden_dim)


        self.pssm_transform = nn.Linear(input_dims['pssm'], hidden_dim)


        self.llm_transform = nn.Linear(input_dims['llm'], hidden_dim)


        self.gcn_ipr_1 = GCNConv(hidden_dim, hidden_dim)
        self.gat_ipr = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=1)
        self.gcn_ipr_2 = GCNConv(hidden_dim, hidden_dim)


        self.gcn_pssm_1 = GCNConv(hidden_dim, hidden_dim)
        self.gat_pssm = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=1)
        self.gcn_pssm_2 = GCNConv(hidden_dim, hidden_dim)


        self.gcn_llm_1 = GCNConv(hidden_dim, hidden_dim)
        self.gat_llm = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=1)
        self.gcn_llm_2 = GCNConv(hidden_dim, hidden_dim)


        self.cnn_ipr = nn.Conv2d(in_channels=2,
                                 out_channels=hidden_dim,
                                 kernel_size=(hidden_dim, 1),
                                 stride=1,
                                 bias=True)
        self.cnn_pssm = nn.Conv2d(in_channels=2,
                                  out_channels=hidden_dim,
                                  kernel_size=(hidden_dim, 1),
                                  stride=1,
                                  bias=True)
        self.cnn_llm = nn.Conv2d(in_channels=2,
                                 out_channels=hidden_dim,
                                 kernel_size=(hidden_dim, 1),
                                 stride=1,
                                 bias=True)


        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )


        self.predict = nn.Linear(hidden_dim, output_dim)


        self.dropout = nn.Dropout(dropout)


        self.skip_weight_ipr = nn.Parameter(torch.tensor(0.8))
        self.skip_weight_pssm = nn.Parameter(torch.tensor(0.8))
        self.skip_weight_llm = nn.Parameter(torch.tensor(0.8))

        self.ipr_importance = nn.Parameter(torch.ones(1) * 0.33)  # Initial equal weights
        self.pssm_importance = nn.Parameter(torch.ones(1) * 0.33)
        self.llm_importance = nn.Parameter(torch.ones(1) * 0.33)


        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):

                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)


        for module in self.modules():
            if isinstance(module, GATConv):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, data, features_dict=None):


        if isinstance(data, Data):
            edge_index = data.edge_index
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None


            if features_dict is None and hasattr(data, 'ipr') and hasattr(data, 'pssm') and hasattr(data, 'llm'):
                features_dict = {
                    'ipr': data.ipr,
                    'pssm': data.pssm,
                    'llm': data.llm
                }
        else:

            edge_index = data
            edge_attr = None


        if features_dict is None:
            raise ValueError("Must provide features_dict or use a Data object containing features")


        h_ipr = self.ipr_transform(features_dict['ipr'])
        h_ipr_gcn1 = torch.relu(self.gcn_ipr_1(h_ipr, edge_index, edge_attr) + h_ipr)
        h_ipr_gat = torch.relu(self.gat_ipr(h_ipr_gcn1, edge_index, edge_attr) + h_ipr_gcn1)
        h_ipr_gcn2 = torch.relu(self.gcn_ipr_2(h_ipr_gat, edge_index, edge_attr) + h_ipr_gat)


        h_pssm = self.pssm_transform(features_dict['pssm'])
        h_pssm_gcn1 = torch.relu(self.gcn_pssm_1(h_pssm, edge_index, edge_attr) + h_pssm)
        h_pssm_gat = torch.relu(self.gat_pssm(h_pssm_gcn1, edge_index, edge_attr) + h_pssm_gcn1)
        h_pssm_gcn2 = torch.relu(self.gcn_pssm_2(h_pssm_gat, edge_index, edge_attr) + h_pssm_gat)


        h_llm = self.llm_transform(features_dict['llm'])
        h_llm_gcn1 = torch.relu(self.gcn_llm_1(h_llm, edge_index, edge_attr) + h_llm)
        h_llm_gat = torch.relu(self.gat_llm(h_llm_gcn1, edge_index, edge_attr) + h_llm_gcn1)
        h_llm_gcn2 = torch.relu(self.gcn_llm_2(h_llm_gat, edge_index, edge_attr) + h_llm_gat)


        X_ipr = torch.cat((h_ipr_gcn1, h_ipr_gcn2), 1).t()
        X_ipr = X_ipr.view(1, 2, self.hidden_dim, -1)

        X_pssm = torch.cat((h_pssm_gcn1, h_pssm_gcn2), 1).t()
        X_pssm = X_pssm.view(1, 2, self.hidden_dim, -1)

        X_llm = torch.cat((h_llm_gcn1, h_llm_gcn2), 1).t()
        X_llm = X_llm.view(1, 2, self.hidden_dim, -1)


        h_ipr_embedding = self.cnn_ipr(X_ipr)
        h_ipr_embedding = h_ipr_embedding.view(self.hidden_dim, -1).t()

        h_pssm_embedding = self.cnn_pssm(X_pssm)
        h_pssm_embedding = h_pssm_embedding.view(self.hidden_dim, -1).t()

        h_llm_embedding = self.cnn_llm(X_llm)
        h_llm_embedding = h_llm_embedding.view(self.hidden_dim, -1).t()


        h_ipr_embedding = self.dropout(h_ipr_embedding)
        h_pssm_embedding = self.dropout(h_pssm_embedding)
        h_llm_embedding = self.dropout(h_llm_embedding)


        raw_weights = torch.cat([self.ipr_importance, self.pssm_importance, self.llm_importance])
        feature_weights = F.softmax(raw_weights, dim=0)


        h_weighted = (h_ipr_embedding * feature_weights[0].view(1, 1) +
                      h_pssm_embedding * feature_weights[1].view(1, 1) +
                      h_llm_embedding * feature_weights[2].view(1, 1))


        h_combined = torch.cat([h_ipr_embedding, h_pssm_embedding, h_llm_embedding], dim=1)
        h_cat_fused = self.fusion(h_combined)


        h_fused = h_cat_fused * 0.7 + h_weighted * 0.3


        logits = self.predict(h_fused)

        return logits
