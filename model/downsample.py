import torch
import torch.nn as nn
from torch import empty, cat

class DownSampleForLSTM(nn.Module):
    """Downsample Module before LSTM

    Downsample module before LSTM.

    Args:
        input_size: length of the input temperature.
        lstm_num_square: Number of lstms is 
                        ``lstm_num_square x lstm_num_square``.
                        Default 3.
        version: default 1.

    Inputs:
        _input: input tensor should be in shape of
        (batch, channel, seq_len, input_size, input_size).
        NOTE: currently all version will only take channel 0 only!
        
    Ouputs:
        downsample temperature grid in shape of
        (seq_len, batch, lstm_num_square, lstm_num_square, version)

    """
    def __init__(self,
                input_size:int,
                lstm_num_square:int=3,
                version:int=3):
        super(DownSampleForLSTM, self).__init__()
        self.lstm_num_square = lstm_num_square
        self.output_channels = version
        self.version = version
        self.pools = nn.ModuleList()
        self.pools.append(SelectCenter(input_size, lstm_num_square))
        if version > 1:
            self.pools.append(nn.AdaptiveAvgPool2d(lstm_num_square))
        if version > 2:
            self.pools.append(nn.AdaptiveMaxPool2d(lstm_num_square))


    def forward(self, _input):
        # Get channel 0 and reshape the input to
        # (seq_len, batch, input_size, input_size) for downsampling
        __input = _input[:,0].permute(1,0,2,3)
        # Create empty output tensor and shape
        _new_shape = list(__input.shape)
        _new_shape[2] = _new_shape[3] = self.lstm_num_square
        output = empty(_new_shape + [0])
        if torch.cuda.is_available():
            output = output.cuda()
        # Compute each downsample tensor and concat them in the new axis 4
        for pool in self.pools:
            pool_out = pool(__input)
            # print(pool)
            # print(pool_out.shape)
            pool_out = pool_out.reshape(_new_shape + [1])
            output = cat([output, pool_out], axis=-1)
        return output


class SelectCenter(nn.Module):
    """Customized sampling module

    Just roughly select the center grid of each block as downsample

    Inputs:
        (seq_len, batch, input_size, input_size)

    Outputs:
        (seq_len, batch, lstm_num_square, lstm_num_square)
    """
    def __init__(self, input_size:int, lstm_num_square:int=3):
        super(SelectCenter, self).__init__()
        # calculate the index to be select
        grid_len = input_size // lstm_num_square
        self.grid_select = []
        for x in range(grid_len // 2, input_size, grid_len):
            self.grid_select.append(x)
        self.grid_select = torch.tensor(self.grid_select)
        if torch.cuda.is_available():
            self.grid_select = self.grid_select.cuda()
            

    def forward(self, _input):
        _output = _input.index_select(-1, self.grid_select)
        return _output.index_select(-2, self.grid_select)
