import torch
import torch.nn as nn


def LossFactory(name: str, device: torch.device):
    if name == 'rmse':
        return RMSELoss()
    elif name == 'mse':
        return nn.MSELoss()
    elif name == 'l1':
        return nn.L1Loss()
    elif name == 'score':
        return NegativeScore(device)
    else:
        raise TypeError('Invalid loss function type')


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self,yhat,y):
        """
        tensor shape [batch, timestep]
        """
        return torch.sqrt(((yhat-y)**2).sum(dim=0) + self.eps).sum()


class NegativeScore(nn.Module):
    def __init__(self, device: torch.device):
        import numpy as np
        super().__init__()
        self.RMSE = RMSELoss()
        self.ln = torch.tensor(
            data=np.log(np.arange(1, 25))\
                *np.array([1.5]*4+[2]*7+[3]*7+[4]*6)\
                *2/3,
            dtype=torch.float,
            device=device,
            requires_grad=False
        )

    def forward(self, yhat, y):
        assert yhat.dim() == 2 and tuple(yhat.shape) == tuple(y.shape) and yhat.shape[1] >= 24
        p = yhat[:, -24:]
        r = y[:, -24:]
        rmse = self.RMSE(p, r)
        mp = torch.mean(p, 0, True)
        mr = torch.mean(r, 0, True)
        fz = ((p-mp)*(r-mr)).sum(0)
        fm = torch.sqrt(((p-mp)**2).sum(0)+((r-mr)**2).sum(0))
        return rmse - (self.ln*fz/fm).sum()
