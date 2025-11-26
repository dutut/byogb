import os.path as osp        #用于处理文件路径
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score  #用于计算roc_auc_score和TSNE等。
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
from torch_geometric.utils import negative_sampling, coalesce
from torch_geometric.datasets import Planetoid                    #专门处理图数据的PyTorch库，
import torch_geometric.transforms as T   #提供了图神经网络（GNN）的一些功能，例如GATConv（图注意力层）、图数据集和图操作函数。
from torch_geometric.nn import GATConv
import torch.nn as nn
from torchmetrics.functional import pairwise_cosine_similarity #用于计算指标，如pairwise_cosine_similarity，计算图嵌入之间的相似度。
from utils import *

device = torch.device('cuda:0' if(torch.cuda.is_available()) else 'cpu')

class GAT(nn.Module):
    def __init__(self, F, H, C, n_head, task, dropout):   #F: 输入特征的维度。H: 中间层（隐藏层）的特征维度。C: 输出类别的数量。n_head: 每层图注意力网络的头数。task: 任务类型，可能影响模型的行为（例如分类或回归）。
        super(GAT, self).__init__()
        self.task = task
        self.conv1 = GATConv(F, H, heads=n_head)        #conv1 和 conv2 是两个图卷积层（GATConv）。dropout: Dropout比例，用于正则化，避免过拟合。
        self.conv2 = GATConv(H * n_head, H, heads=n_head)#conv1 的输入维度为 F，输出维度为 H，且使用 n_head 个注意力头；conv2 的输入维度为 H * n_head，输出为 H。
        self.classifier = nn.Linear(H * n_head, C)     #classifier 是一个全连接层，输出的维度为 C，用于最终的分类任务。
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.classifier(x)
        return x
          
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        #x = x.relu()
        return x
    
    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        logits = cos(z[edge_index[0]], z[edge_index[1]])
        logits = (logits+1)/2
        return logits

    def decode_target(self, z, edge_index, target, buffer, knn, thres):
        #finding added edges
        prob_adj = pairwise_cosine_similarity(z[buffer,:].reshape(-1,z.size(1)),z[target,:])
        prob_adj = (prob_adj+1)/2

        connected = find_connected_nodes(buffer, edge_index)
        mask = torch.ones(prob_adj.size(1)).to(device)
        for i, v in enumerate(connected):
            if v in target:
                indices = torch.nonzero(target==v).squeeze().item()
                mask[indices] = 0
        mask = mask.reshape(-1,mask.size(0))
        valid_prob_adj = torch.mul(prob_adj, mask)

        v, i = valid_prob_adj.topk(knn)
        i = torch.tensor([target[item] for item in i[0]]).to(device)
        i = i.reshape(-1, i.size(0))
        buffer_index = torch.full((i.size(1),),buffer).to(device)
        buffer_index = buffer_index.reshape(-1,buffer_index.size(0))
        topk_index = torch.cat((i, buffer_index), dim=0)
        topk_index_inv = torch.stack((topk_index[1], topk_index[0]), dim=0)
        valid_edge_index = coalesce(torch.cat((topk_index, topk_index_inv), dim=-1))
        
        delete_indices = []
        target_edge_index = extract_edges_with_node(edge_index, buffer)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for i in range(target_edge_index.size(1)):
            
            output = cos(z[target_edge_index[0][i],:].reshape(-1,z.size(1)), z[target_edge_index[1][i],:].reshape(-1,z.size(1))).item()
            if output <= 0.7:
                delete_indices.append(i)
            
        deleted_edge_index = target_edge_index[:,delete_indices]

        return valid_edge_index, deleted_edge_index