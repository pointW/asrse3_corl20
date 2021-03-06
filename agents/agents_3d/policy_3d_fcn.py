import numpy as np
import torch
import torch.nn.functional as F
from agents.agents_3d.dqn_3d_fcn import DQN3DFCN

class Policy3DFCN(DQN3DFCN):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7*np.pi/8)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)

    def update(self, batch):
        self._loadBatchToDevice(batch)
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        output = self.forwardFCN(states, obs[1], obs[0])
        output = output.reshape(batch_size, -1)
        target = action_idx[:, 2] * self.heightmap_size * self.heightmap_size + \
                 action_idx[:, 0] * self.heightmap_size + \
                 action_idx[:, 1]
        loss = F.cross_entropy(output, target)

        self.fcn_optimizer.zero_grad()
        loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        self.loss_calc_dict = {}

        return loss.item(), torch.tensor(0.)


