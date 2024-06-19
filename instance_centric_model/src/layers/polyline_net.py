import torch
import torch.nn as nn
import torch.nn.functional as F


class PolylineNet(nn.Module):
    def __init__(self, input_size, hidden_size, out_size=None):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        
        if out_size is not None:
            self.fc_out = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, out_size, bias=True)
            )
        else:
            self.fc_out = None 
        

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C): B,N,20,13
            polylines_mask (batch_size, num_polylines, num_points_each_polylines): B,N,20

        Returns:
            B,N,D
        """
        bs, poly_num, point_num, C = polylines.shape
        poly_feat_valid = self.fc1(polylines[polylines_mask])  # (N, C)   (B,N,20,13)->(S,13)->(S,d_model)
        poly_feat = polylines.new_zeros(bs, poly_num, point_num, poly_feat_valid.shape[-1])# B,N,20,d_model
        poly_feat[polylines_mask] = poly_feat_valid 
        
        # get global feature
        pooled_feat = poly_feat.max(dim=2)[0] # B,N,d_model  
        poly_feat = torch.cat((poly_feat, pooled_feat[:, :, None, :].repeat(1, 1, point_num, 1)), dim=-1) # B,N,20,d_model + B,N,20,d_model
        # mlp
        poly_feat_valid = self.fc2(poly_feat[polylines_mask]) # B,N,20,2d-> S,2d->S,d
        feat_buffers = poly_feat.new_zeros(bs, poly_num, point_num, poly_feat_valid.shape[-1])# B,N,20,d
        feat_buffers[polylines_mask] = poly_feat_valid
        # max-pooling
        feat_buffers = feat_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)  BNd
        
        # out-mlp 
        if self.fc_out is not None:
            valid_mask = (polylines_mask.sum(dim=-1) > 0) # B,N
            feat_buffers_valid = self.fc_out(feat_buffers[valid_mask])  # (N, C)     S,d->S, 128
            feat_buffers = feat_buffers.new_zeros(bs, poly_num, feat_buffers_valid.shape[-1]) # B,N,128
            feat_buffers[valid_mask] = feat_buffers_valid
        return feat_buffers