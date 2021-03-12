from model.v1 import SolutionV1
from model.v2 import SolutionV2
from model.v3 import Solutionv3
import torch.nn as nn
from torch import tensor


class Solution(nn.Module):
    """
    Final Model Factory

    args:
        version: int
            model version
    """
    def __init__(self, version: int = 2, *args, **kwargs) -> None:
        super(Solution, self).__init__()
        if version == 1:
            self.model = SolutionV1(*args, **kwargs)
        elif version == 2:
            self.model = SolutionV2(*args, **kwargs)
        else:
            raise ValueError('Invalid model version!')

    def forward(self, _input: tensor) -> tensor:
        return self.model(_input)

    def infer(self, _input: tensor) -> tensor:
        return self.model.infer(_input)
