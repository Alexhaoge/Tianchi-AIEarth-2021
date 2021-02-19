import torch.nn as nn
import torch

from .downsample import DownSampleForLSTM

class SocialLSTM(nn.Module):
    """Social LSTM

    Args:

    Inputs:
        _input: input tensor should be in shape of
        (batch, channel, seq_len,  input_size, input_size).
        NOTE: currently all version will only take channel 0 only!
            
    Returns:
        output of cells
        (seq_len, batch, lstm_num_square, lstm_num_square)
    """

    def __init__(self, input_size:int,
                lstm_num_square:int=3,
                embedding_size:int=16,
                hidden_size:int=32,
                # layer_num:int=1,
                dropout_rate:float=0.5,
                downsample_version:int=3,
                upsample:bool=False):
        super(SocialLSTM, self).__init__()
        self.input_size = input_size
        self.lstm_num_square = lstm_num_square
        self.hidden_size = hidden_size
        # Downsample layer before LSTM
        self.downsample = DownSampleForLSTM(input_size, lstm_num_square)
        # Create lstm grids
        self.cells = nn.ModuleList()
        # self.linear_io = nn.ModuleList()
        self.linear_ho = nn.ModuleList()
        for _ in range(lstm_num_square):
            lstm_row = nn.ModuleList()
            # io_row = nn.ModuleList()
            ho_row = nn.ModuleList()
            for _ in range(lstm_num_square):
                lstm_row.append(nn.LSTMCell(
                    input_size = embedding_size * 2,
                    hidden_size = hidden_size
                ))
                # io_row.append(nn.Linear(embedding_size*2, hidden_size))
                ho_row.append(nn.Linear(hidden_size, hidden_size))
            self.cells.append(lstm_row)
            # self.linear_io.append(io_row)
            self.linear_ho.append(ho_row)
        
        # LINEAR embedding layer after downsample
        self.down_linear_embed = nn.Linear(downsample_version, embedding_size)
        # LINEAR embedding layer for social tensor
        self.social_linear_embed = nn.Linear((lstm_num_square**2)*hidden_size, embedding_size)
        # activation and dropout unit
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        # output FC layer
        self.out_linear = nn.Linear(hidden_size, 1)

           
    def forward(self, _input:torch.Tensor,
                hidden_states:torch.Tensor=None,
                cell_states:torch.Tensor=None):
        '''
        Forward pass for the model
        '''
        down_embedded = self.dropout(self.relu(
            self.down_linear_embed(self.downsample(_input))))
        _shape = list(down_embedded.shape)
        _shape[2] = _shape[3] = self.lstm_num_square
        _shape[4] = self.hidden_size
        __shape = _shape[1:]
        
        outputs = torch.empty(_shape)
        
        if hidden_states == None:
            hidden_states = torch.zeros(__shape)
        if cell_states == None:
            cell_states = torch.zeros(__shape)

        if torch.cuda.is_available():
            outputs = outputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()

        # For each frame in the sequence
        for _t, frame in enumerate(down_embedded):            
            # Compute the social tensor
            social_tensor = hidden_states.view(-1, self.lstm_num_square*self.lstm_num_square*self.hidden_size)
            # Embed the social tensor
            social_embeded = self.dropout(self.relu(self.social_linear_embed(social_tensor)))
            
            _hiddens = torch.empty(__shape)
            _cells = torch.empty(__shape)
            if torch.cuda.is_available():
                _hiddens = _hiddens.cuda()
                _cells = _cells.cuda()
            
            # Throw all those tensors into lstm cell
            for i in range(self.lstm_num_square):
                for j in range(self.lstm_num_square):
                    # concat embedded downsample and social tensor
                    concat_embedded = torch.cat((frame[:, i, j, :], social_embeded), 1)
                    # get lstm cell output
                    _hidden, _cell = self.cells[i][j](
                        concat_embedded,
                        (hidden_states[:, i, j, :], cell_states[:, i, j, :])
                    )
                    outputs[_t, :, i, j, :] = self.sigmoid(
                        self.linear_ho[i][j](_hidden)
                        # self.linear_io[i][j](concat_embedded)
                    )
                    _hiddens[:, i, j, :] = _hidden
                    _cells[:, i, j, :] = _cell
            hidden_states = _hiddens
            cell_states = _cells

        return self.out_linear(outputs).squeeze()
