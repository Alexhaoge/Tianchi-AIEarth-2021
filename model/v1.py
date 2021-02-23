from model.convlstm import ConvLSTM
import torch.nn as nn
import torch


class SolutionV1(nn.Module):
    """Model Version 1
    ConvLSTM -> avgpool2d on each time ->  dropout(linear) 

    Args:

    Inputs:
        _input: input tensor should be in shape of
        (batch, 36, 4, 24, 72).
            
    Returns:
        (batch, 24)
    """

    def __init__(
        self,
        device,
        kernel_size: tuple = (3, 3),
        hidden_dim: int = 16,
        pool_kernel: tuple = (8, 8),
        num_layers: int = 1,
        dropout_rate: float = 0.3,
    ):
        super(SolutionV1, self).__init__()
        self.convlstm = ConvLSTM(4, hidden_dim, kernel_size, num_layers,
                                 batch_first=True, bias=True,
                                 return_all_layers=False)
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.AvgPool2d(pool_kernel)
        self.fc = nn.Linear(
            in_features=(24//pool_kernel[0])*(72//pool_kernel[1])*hidden_dim,
            out_features=2
        )

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        out = self.convlstm(_input[:, :12, :, :, :])[
            0][-1]  # (B, 12, hidden_dim, 24, 72)
        out2 = torch.empty((
            out.shape[0], out.shape[1], out.shape[2],
            out.shape[3]//self.pool.kernel_size[0],
            out.shape[4]//self.pool.kernel_size[1]
        ), device=self.device)
        for i in range(out.shape[1]):
            out2[:, i, :, :, :] = self.pool(out[:, i, :, :, :])
        # -> (B, 12, hidden_dim, 24//pk[0], 72//pk[1])
        out2 = out2.flatten(start_dim=2)  # (B, 12, -1)
        out2 = self.dropout(self.fc(out2))  # (B, 12, 2)
        return out2.flatten(start_dim=1)  # (B, 24)

    def infer(self, _input: torch.Tensor) -> torch.Tensor:
        assert _input.shape[1] == 12 and _input.dim() == 5
        return self.forward(_input)
