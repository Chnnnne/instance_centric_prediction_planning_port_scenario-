import torch
import torch.nn as nn
import torch.nn.functional as F
from .polyline_net import PolylineNet

class PlanNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.plan_mlp = PolylineNet(input_size, hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size*2, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 1), 
            nn.Sigmoid())
        self.dropout =  nn.Dropout(0.3)

    def forward(self, agent_feats, plan_traj, plan_traj_mask):
        """
        plan_traj: B,N,d,2
        plan_traj_mask: B,N,d
        """
        batch_size, agent_num, _ = agent_feats.size() 
        ego_feat = agent_feats[:, 0].unsqueeze(1).expand(-1, agent_num, -1)
        gate_feat = torch.cat((agent_feats, ego_feat), dim=-1)
        gate = self.gate(gate_feat).squeeze(-1) # B*N
        
        if plan_traj.dim() == 3:
            plan_traj = plan_traj.unsqueeze(1)
            plan_traj_mask = plan_traj_mask.unsqueeze(1)
        plan_feat = self.dropout(self.plan_mlp(plan_traj, plan_traj_mask)) # B*N*D
        plan_feat = torch.einsum('bnd,bn->bnd', plan_feat, gate)
        agent_feats = agent_feats + plan_feat
        return agent_feats, gate
