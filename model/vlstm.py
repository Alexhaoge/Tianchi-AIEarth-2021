from torch import tensor
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from .downsample import DownSampleForLSTM
import torch

class VanillaLSTM(nn.Module):
    """Vanilla LSTMs

    Simple vanilla LSTMs for baseline. After downsample, 
    Each LSTM is responsible for a block and runs independently.

    Args:

    Inputs:
        _input: input tensor should be in shape of
        (batch, channel, seq_len,  input_size, input_size).
        NOTE: currently all version will only take channel 0 only!
        
    Returns:
        output of LSTMs
        (seq_len, batch, lstm_num_square, lstm_num_square)
    """
    def __init__(self,
                input_size:int,
                lstm_num_square:int=3,
                embedding_size:int=16,
                hidden_size:int=32,
                layer_num:int=1,
                dropout_rate:float=0.5,
                downsample_version:int=3,
                upsample:bool=False):
        super(VanillaLSTM, self).__init__()
        self.input_size = input_size
        self.lstm_num_square = lstm_num_square
        self.hidden_size = hidden_size
        # Downsample layer before LSTM
        self.downsample = DownSampleForLSTM(
            input_size, lstm_num_square, downsample_version)
        # Create lstm grids
        self.lstms = nn.ModuleList()
        for _ in range(lstm_num_square):
            lstm_row = nn.ModuleList()
            for _ in range(lstm_num_square):
                lstm_row.append(nn.LSTM(
                    input_size = embedding_size,
                    hidden_size = hidden_size,
                    num_layers = layer_num,
                    dropout=dropout_rate))
            self.lstms.append(lstm_row)
        # LINEAR embedding layer after downsample
        self.down_linear_embed = nn.Linear(downsample_version, embedding_size)
        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        # output FC layer
        self.out_linear = nn.Linear(hidden_size, 1)
        

    def forward(self, _input):
        downsample_out = self.downsample(_input)
        output_shape = list(downsample_out.shape)
        output_shape.pop()
        output = torch.empty(output_shape)
        if torch.cuda.is_available():
            output = output.cuda()
        for i in range(self.lstm_num_square):
            for j in range(self.lstm_num_square):
                # embed the output of downsample
                _x = self.dropout(self.relu(
                    self.down_linear_embed(downsample_out[:, :, i, j, :])))
                # throw the embeded _x into lstm
                _tmp = self.lstms[i][j](_x)[0]
                # get temperature by linear layer
                output[:, :, i, j] = self.out_linear(_tmp).squeeze()
        return output