import torch.nn.functional as F
from torch import nn
from tcn_mnist import TemporalConvNet
import convlstm


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, nhid, kernel_size, dropout):
        super(TCN, self).__init__()
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.layer1 = convlstm.ConvLSTM(input_size, num_channels, input_size, nhid, kernel_size, 1, dropout)  #总层数，2分片，1级联层
            # nn.LSTM(1, nhid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = inputs.squeeze().unsqueeze(2)  # .cuda()
        y, _ = self.layer1(x)
        y = y.permute(0, 2, 1)  # .cuda()
        z = self.dropout(y[:, :, -1])
        o = self.linear(z)
        return F.log_softmax(o, dim=1)

        # x = inputs.squeeze().unsqueeze(2)  # .cuda()
        # y, _ = self.layer1(x)
        # y = y[0]  # .squeeze()
        # y = y.permute(0, 2, 1)  # .cuda()
        # o = self.linear(y[:, :, -1])
        # return F.log_softmax(o, dim=1)

        # for TCN only in the top layer
        # x = inputs.squeeze().unsqueeze(2)  # .cuda()
        # y, _ = self.layer1(x)
        # o = self.linear(y)
        # return F.log_softmax(o, dim=1)

        # y = self.dropout(y)
        # y1 = self.tcn(y)  # input should have dimension (N, C, L)
        # o = self.linear(y1[:, :, -1])
        # return F.log_softmax(o, dim=1)
