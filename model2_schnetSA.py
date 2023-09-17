# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from torch.nn.functional import selu
from SchNet_SA import SchNet_atten
import random

import numpy as np
def generate_decreasing_sequence(n, alpha=2.0):
    k = np.arange(1, n+1)
    x = 1 / np.power(k, alpha)
    x = x / np.sum(x)
    return x

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        #print (z.size())
        beta = torch.softmax(w, dim=1)
        #print (beta.size)
        return (beta * z).sum(1)


class GNNLayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout,GAT_Layers,W_size,Nei_number=10):

        super(GNNLayer, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        self.gat_layers1 = nn.ModuleList()
        self.gat_layers2 = nn.ModuleList()
        self.w_h = nn.Linear(in_size,W_size,bias=False)
        self.nums_GAT=GAT_Layers
        self.Nei_number=Nei_number
    ##############3_layers GAT
        if GAT_Layers==3:
            for i in range(len(meta_paths)):
                self.gat_layers.append(GATConv_mine(W_size, out_size, layer_num_heads, dropout, 0, activation=F.elu, residual=False, Nei_number=self.Nei_number))
            for i in range(len(meta_paths)):
                self.gat_layers1.append(GATConv_mine(out_size*layer_num_heads, out_size, layer_num_heads, dropout, 0, activation=F.elu, Nei_number=self.Nei_number))
            for i in range(len(meta_paths)):
                self.gat_layers2.append(GATConv_mine(out_size*layer_num_heads, out_size, layer_num_heads, dropout, 0, activation=F.elu, Nei_number=self.Nei_number))


################################2-layerGAT
        if GAT_Layers==2:
            for i in range(len(meta_paths)):
                self.gat_layers.append(GATConv_mine(W_size, out_size, layer_num_heads, dropout, 0, activation=F.elu, residual=False, Nei_number=self.Nei_number))
            for i in range(len(meta_paths)):
                self.gat_layers1.append(GATConv_mine(out_size*layer_num_heads, out_size, layer_num_heads, dropout, 0, activation=F.elu, Nei_number=self.Nei_number))

########1_layer
        if GAT_Layers==1:
            for i in range(len(meta_paths)):
                self.gat_layers.append(GATConv_mine(W_size,out_size,layer_num_heads,dropout, 0, activation=F.elu,residual=False,Nei_number=self.Nei_number))
            #self.gat_layers.append(self.num_GAT(in_size,out_size))


        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            w_h=self.w_h(h) # 256dim
            if self.nums_GAT==3:
                decay_xishu = generate_decreasing_sequence(3, 1.5)
                a = self.gat_layers[i](new_g, w_h).flatten(1)
                b = self.gat_layers1[i](new_g, a).flatten(1)
                c = self.gat_layers2[i](new_g, b).flatten(1)

                semantic_embeddings.append(a * decay_xishu[0] + b * decay_xishu[1] + c*decay_xishu[2])

            if self.nums_GAT==2:
                decay_xishu = generate_decreasing_sequence(2,1.5)
                a = self.gat_layers[i](new_g, w_h).flatten(1)
                b = self.gat_layers1[i](new_g, a).flatten(1)

                semantic_embeddings.append(a * decay_xishu[0] + b * decay_xishu[1])

            if self.nums_GAT==1:
                semantic_embeddings.append(self.gat_layers[i](new_g,w_h).flatten(1))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)
        #print (self.semantic_attention(semantic_embeddings).shape)
        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)



class GNN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout,GAT_Layers,W_size,Nei_number):
        super(GNN, self).__init__()
        self.Nei_number=Nei_number
        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(meta_paths, in_size, hidden_size, num_heads[0],dropout,GAT_Layers,W_size,self.Nei_number))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size,bias=False)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
            print (h.shape)
            print (self.predict(h).shape)
        return self.predict(h)



class HiMGNN(nn.Module):

    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, num_heads, dropout,GAT_Layers,W_size,Nei_number,lamda,cutoff):

        super(HiMGNN, self).__init__()
        self.sum_layers = nn.ModuleList()
        self.drug_3D_layers = SchNet_atten(energy_and_force=False, cutoff=cutoff, num_layers=1,
                             hidden_channels=32, num_filters=16, num_gaussians=5,
                             out_channels=16).to('cpu')

        self.in_size = in_size

        self.lamda = lamda
        self.Nei_number=Nei_number

        self.Wd = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(16, 16 * 2, bias=True),
            nn.ReLU(),
            nn.Linear(16 * 2, 64, bias=True)
        )

        self.fusion = nn.Linear(64,in_size[0], bias=False)
        self.fusion2 = nn.Linear(64, in_size[1], bias=False)

        for i in range(0,len(all_meta_paths)):

            self.sum_layers.append(GNN(all_meta_paths[i], in_size[i], hidden_size, out_size, num_heads, dropout,GAT_Layers,W_size,self.Nei_number))

        self.bilide = BilinearDecoder(feature_size=out_size)


    def forward(self,s_g, pos_3d,s_h_1,s_h_2,noise):

        # print (self.layers)

        tem = self.drug_3D_layers(pos_3d.to('cpu').to('cuda:0'))
        s_h_1_3d = self.Wd(tem)
        s_h_1 = s_h_1 * (1 - self.lamda[0]) + s_h_1_3d * self.lamda[0]
        h1 = self.sum_layers[0](s_g[0], s_h_1)
        h2 = self.sum_layers[1](s_g[1], s_h_2)

        return h1, h2, self.bilide(h1, h2.t())



class BilinearDecoder(nn.Module):
    def __init__(self, feature_size=64):
        super(BilinearDecoder, self).__init__()
        self.Q = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            # nn.ReLU()
        )

        self.jihuo = nn.Sigmoid()


    def forward(self, h_d, h_vir):
        d = self.Q(h_d)
        results_mask=torch.mm(d, h_vir)
        results_mask=self.jihuo(results_mask)
        return results_mask



import dgl.function as fn
from dgl import DGLError
# from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import edge_softmax
import torch as th
from torch import nn

class GATConv_mine(GATConv):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 Nei_number=10):
        super(GATConv_mine, self).__init__(in_feats=in_feats,
                 out_feats=out_feats,
                 num_heads=num_heads,
                 feat_drop=feat_drop,
                 attn_drop=attn_drop,
                 negative_slope=negative_slope,
                 residual=residual,
                 activation=activation)

        self.N_number = Nei_number

    def reset_parameters(self):
        super(GATConv_mine, self).reset_parameters()

    def forward(self, graph, feat):

        graph = graph.local_var()
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))

        if self.N_number>0:

            temp_attn = self.attn_drop(edge_softmax(graph, e))
            k = min(self.N_number, temp_attn.shape[0])
            topk_scores, topk_indices = th.topk(temp_attn, k=k, dim=0)
            topk_scores_norm = th.nn.Softmax(dim=0)(topk_scores)
            temp_attn_new = th.zeros_like(temp_attn)
            temp_attn_new.scatter_(0, topk_indices, topk_scores_norm)
            graph.edata['a'] = temp_attn_new

            # temp_attn = self.attn_drop(edge_softmax(graph, e))
            # edata_zeros = th.zeros(graph.number_of_edges(), temp_attn.shape[1], temp_attn.shape[2]).to('cuda:0')
            # k = min(self.N_number, temp_attn.shape[0])
            # topk = th.topk(temp_attn, k=k, dim=0)[1]
            # softmax_topk_e = th.randn(self.N_number, temp_attn.shape[1], temp_attn.shape[2]).to('cuda:0')
            # for j in range(temp_attn.shape[1]):
            #     for i in range(self.N_number):
            #         softmax_topk_e[i, j] = temp_attn[topk[i, j, 0]][j]
            # softmax_topk_e = th.nn.Softmax(dim=0)(softmax_topk_e)
            # for j in range(temp_attn.shape[1]):
            #     for i in range(self.N_number):
            #         edata_zeros[topk[i, j, 0]][j] = softmax_topk_e[i, j]
            # graph.edata['a'] = edata_zeros

#########################################################

            # graph.edata['b'] = torch.zeros(graph.edata['a'].shape)+0.0000000000001
            # a_temp = graph.edata['a'].view(graph.edata['a'].shape[0], -1)
            # nodesN = graph.edges()[0]
            # for ii in range(a_temp.shape[1]):  # head
            #     for ff in tqdm(range(feat.shape[0])):  # node
            #         index_x = []
            #         index_y = []
            #         for ni in range(len(nodesN)):
            #             if ff == nodesN[ni]:
            #                 index_x.append(ni)
            #                 index_y.append(ii)
            #         value, index = torch.sort(a_temp[index_x, index_y], descending=True)
            #         if self.N_number > len(index):
            #             N_number = len(index)
            #         else:
            #             N_number = self.N_number
            #         value_guiyi = nn.Softmax(dim=0)(value[:N_number])
            #         for jj in range(N_number):
            #             graph.edata['b'][index_x[index[jj]]][ii][0] = value_guiyi[jj]
            #
            # # for ii in range(a_temp.shape[1]):
            # #     value, index = torch.sort(a_temp[:,ii], descending=True)
            # #     value_guiyi = nn.Softmax(dim=0)(value[:self.N_number])
            # #     for jj in range(self.N_number):
            # #         graph.edata['b'][index[jj]][ii][0]=value_guiyi[jj]
            # graph.edata['b']=graph.edata['b'].to('cuda:0')
            # graph.update_all(fn.u_mul_e('ft', 'b', 'm'),
            #                  fn.sum('m', 'ft'))
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
        else:
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))  # 根据所有的边graph.edges()，把节点i对应所有邻居的边权进行softmax
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))  # max,min,mean需要修改相应的边权重

        # graph.update_all(message_func=mes_func, reduce_func=red_func)

        rst = graph.dstdata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst


def mes_func(edges):
    # print(edges.src['ft'].shape)
    # print(edges.dst['ft'].shape)
    # print(edges.data['a'].shape)
    return {'m' : edges.src['ft']*edges.data['a']}

def red_func(nodes):
    # print(nodes.mailbox['m'].shape)
    return {'ft' : nodes.mailbox['m'].sum(dim=1)}


