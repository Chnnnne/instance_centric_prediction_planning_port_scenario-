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
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
        bs, poly_num, point_num, C = polylines.shape
        poly_feat_valid = self.fc1(polylines[polylines_mask])  # (N, C)
        poly_feat = polylines.new_zeros(bs, poly_num, point_num, poly_feat_valid.shape[-1])
        poly_feat[polylines_mask] = poly_feat_valid
        
        # get global feature
        pooled_feat = poly_feat.max(dim=2)[0]
        poly_feat = torch.cat((poly_feat, pooled_feat[:, :, None, :].repeat(1, 1, point_num, 1)), dim=-1)
        # mlp
        poly_feat_valid = self.fc2(poly_feat[polylines_mask])
        feat_buffers = poly_feat.new_zeros(bs, poly_num, point_num, poly_feat_valid.shape[-1])
        feat_buffers[polylines_mask] = poly_feat_valid
        # max-pooling
        feat_buffers = feat_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)
        
        # out-mlp 
        if self.fc_out is not None:
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feat_buffers_valid = self.fc_out(feat_buffers[valid_mask])  # (N, C)
            feat_buffers = feat_buffers.new_zeros(bs, poly_num, feat_buffers_valid.shape[-1])
            feat_buffers[valid_mask] = feat_buffers_valid
        return feat_buffers