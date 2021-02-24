import numpy as np
import torch
from agents.base_agent import BaseAgent

class Base3D(BaseAgent):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7*np.pi/8)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size)

        self.num_rz = num_rz
        self.rzs = torch.from_numpy(np.linspace(rz_range[0], rz_range[1], num_rz)).float()

    def getActionFromPlan(self, plan):
        x = plan[:, 0:1]
        y = plan[:, 1:2]
        rot = plan[:, 2:3]
        states = plan[:, 3:4]
        pixel_x = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_y = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()
        pixel_x = torch.clamp(pixel_x, 0, self.heightmap_size-1)
        pixel_y = torch.clamp(pixel_y, 0, self.heightmap_size-1)
        rot_id = (rot.expand(-1, self.num_rz) - self.rzs).abs().argmin(1).unsqueeze(1)

        x = (pixel_x.float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixel_y.float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        rot = self.rzs[rot_id]
        actions = torch.cat((x, y, rot), dim=1)
        action_idx = torch.cat((pixel_x, pixel_y, rot_id), dim=1)
        return action_idx, actions