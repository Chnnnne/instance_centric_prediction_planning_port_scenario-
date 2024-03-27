import torch
import torch.nn as nn
import torch.nn.functional as F

class SftLayer(nn.Module):
    def __init__(self, d_edge, d_model, d_ffn, n_head = 8, dropout = 0.1, update_edge = True):
        super().__init__()
        self.update_edge = update_edge

        self.proj_memory = nn.Sequential(
            nn.Linear(d_model + d_model + d_edge, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        if self.update_edge:
            self.proj_edge = nn.Sequential(
                nn.Linear(d_model, d_edge),
                nn.LayerNorm(d_edge),
                nn.ReLU(inplace=True)
            )
            self.norm_edge = nn.LayerNorm(d_edge)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=False)

        # Feedforward model
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, node, edge):
        # update node
        x, edge, memory = self._build_memory(node, edge)
        x_prime, _ = self._mha_block(x, memory, attn_mask=None, key_padding_mask=None)
        x = self.norm2(x + x_prime).squeeze()
        x = self.norm3(x + self._ff_block(x))
        return x, edge

    def _build_memory(self, node, edge):
        n_token = node.shape[0]

        # 1. build memory
        src_x = node.unsqueeze(dim=0).repeat([n_token, 1, 1])  # (N, N, d_model)
        tar_x = node.unsqueeze(dim=1).repeat([1, n_token, 1])  # (N, N, d_model)
        memory = self.proj_memory(torch.cat([edge, src_x, tar_x], dim=-1))  # (N, N, d_model)
        # 2. (optional) update edge (with residual)
        if self.update_edge:
            edge = self.norm_edge(edge + self.proj_edge(memory))  # (N, N, d_edge)

        return node.unsqueeze(dim=0), edge, memory

    # multihead attention block
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x, _ = self.multihead_attn(x, mem, mem,
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False)  # return average attention weights
        return self.dropout2(x), None

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



class FusionNet(nn.Module):
    def __init__(self, d_model, d_edge, n_head=4, n_layers=4, dropout=0.1, update_edge=True):
        super().__init__()
        fusion = []
        for i in range(n_layers):
            need_update_edge = False if i == n_layers - 1 else update_edge
            fusion.append(SftLayer(d_edge=d_edge,
                                   d_model=d_model,
                                   d_ffn=d_model*2,
                                   n_head=n_head,
                                   dropout=dropout,
                                   update_edge=need_update_edge))
        self.fusion = nn.ModuleList(fusion)

    def forward(self, agent_feats, agent_mask, map_feats, map_mask, rpe_feats, rpe_mask):
        """
        agent_feats: batch_size, agent_num, dim
        agent_mask: batch_size, agent_num
        map_feats: batch_size, map_num, dim
        map_mask: batch_size, map_num
        rpe_feats: batch_size, N, N, dim_1
        rpe_mask: batch_size, N, N
        """
        x = torch.cat((agent_feats, map_feats), dim=1) 
        x_mask = torch.cat((agent_mask, map_mask), dim=1)
        batch_size, all_num, dim = x.shape
        agent_num = agent_feats.shape[1]
        
        agents_new, maps_new = list(), list()
        for i in range(batch_size):
            x_frame = x[i]
            x_mask_frame = x_mask[i]
            x_frame = x_frame[x_mask_frame].view(-1, dim) # (valid_num,dim)
            valid_num = x_frame.shape[0]
            rpe_frame = rpe_feats[i][rpe_mask[i]].view(valid_num, valid_num, -1)
            for mod in self.fusion:
                x_frame, rpe_frame= mod(x_frame, rpe_frame)
            out = x_frame.new_zeros(all_num, x_frame.shape[-1])
            out[x_mask_frame] = x_frame
            agents_new.append(out[:agent_num])
            maps_new.append(out[agent_num:])
        agent_feats = torch.stack(agents_new)
        map_feats = torch.stack(maps_new)
        return agent_feats, map_feats