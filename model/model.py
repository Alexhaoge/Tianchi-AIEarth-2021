from model.convlstm import ConvLSTM
import torch.nn as nn
import torch


class Solution(nn.Module):
    """Final Model

    Args:

    Inputs:
        _input: input tensor should be in shape of
        (batch, timestep, 4, 24, 72).
            
    Returns:
        (batch, timestep)
    """

    def __init__(
        self,
        device, 
        kernel_size: tuple = (3,3), 
        hidden_dim: int = 16,
        pool_kernel: tuple = (8, 8),
        num_layers: int = 1,
        dropout_rate: float = 0.3,
    ):
        super(Solution, self).__init__()
        self.convlstm = ConvLSTM(4, hidden_dim, kernel_size, num_layers, 
                                batch_first=True, bias=True, 
                                return_all_layers=False)
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.AvgPool2d(pool_kernel)
        self.fc = nn.Linear(
            in_features=(24//pool_kernel[0])*(72//pool_kernel[1])*hidden_dim,
            out_features=1
        )

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        out = self.convlstm(_input)[0][-1]  # (B, T, hidden_dim, 24, 72)
        out2 = torch.empty((
            out.shape[0], out.shape[1], out.shape[2],
            out.shape[3]//self.pool.kernel_size[0],
            out.shape[4]//self.pool.kernel_size[1]
        ), device=self.device)
        for i in range(out.shape[1]):
            out2[:,i,:,:,:] = self.pool(out[:,i,:,:,:])  
        # -> (B, T, hidden_dim, 24//pk[0], 72//pk[1])
        out2 = out2.flatten(start_dim=2)  # (B, T, -1)
        out2 = self.dropout(self.fc(out2))  # (B, T, 1)
        return out2.squeeze()

    def infer(self, _input: torch.Tensor) -> torch.Tensor:
        assert _input.shape[1] == 12 and _input.dim() == 5
        return self.forward(torch.cat([_input, _input], dim=1))
