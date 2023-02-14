import torch.nn as nn
import torch
from torch import einsum
from tcn_mnist import TemporalConvNet
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # self.kernel_size = kernel_size
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2


        # self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
        #                       out_channels=4 * self.hidden_dim,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=self.bias)
        self.fc = nn.Linear(self.input_dim + self.hidden_dim, 2 * self.hidden_dim, bias=False)
        self.fc2 = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim, bias=False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.w_hh = Parameter(torch.Tensor(1, self.input_dim + self.hidden_dim)).to(self.device)
        self.bias = Parameter(torch.Tensor(1, 1)).to(self.device)
        self.reset_weigths()

        # self.fc3 = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim, bias=False)

    def reset_weigths(self):
        """reset weights
        """
        # for weight in self.parameters():
        #     nn.init.xavier_normal_(weight)
        self.bias = Parameter(torch.tensor(0.0))
        torch.nn.init.xavier_normal_(self.w_hh)

    def forward(self, input_tensor, cur_state):
        h_cur = cur_state

        combined = torch.cat([input_tensor.to(self.device), h_cur.to(self.device)],
                             dim=1)  # concatenate along channel axis

        # combined_conv = self.conv(combined)
        combined_conv = self.fc(combined.squeeze()).to(self.device)

        cc_z, cc_r = torch.split(combined_conv, self.hidden_dim, dim=1)
        z = torch.sigmoid(cc_z).to(self.device)
        r = torch.sigmoid(cc_r).to(self.device)
        rh = r * h_cur.to(self.device)
        conbined2 = torch.cat([input_tensor.to(self.device), rh.to(self.device)], dim=1)
        con2 = torch.tanh(self.fc2(conbined2.squeeze()).to(self.device))

        h_next = (1-z) * h_cur.to(self.device) + z*con2

        # bias = Parameter(torch.Tensor(1, 1)).to(self.device)
        # delta_u = torch.sigmoid(F.linear(combined.squeeze().to(self.device), self.w_hh))   #sigmoid
        # delta_u = torch.sigmoid(F.linear(combined.squeeze().to(self.device), self.w_hh, self.bias))
        delta_u = torch.sigmoid((F.linear(combined.squeeze(), self.w_hh, self.bias))) / 1e2
        # delta_u = torch.sigmoid(F.relu(F.linear(combined.squeeze(), self.w_hh, self.bias))) / 1e4

        # delta_u = torch.sigmoid(F.relu(F.linear(combined.squeeze(), self.w_hh))) / 1e3

        # g = torch.sigmoid(self.fc3(combined.squeeze())).to(self.device)

        return h_next, delta_u

    def init_hidden(self, batch_size):
        # height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, device=self.fc.weight.device) + 1e-9,
                torch.zeros(batch_size, self.hidden_dim, device=self.fc.weight.device) + 1e-9)


def inverse4sigmoid(x):
    y = torch.log(x / (1 - x))
    return y


class Attention(nn.Module):
    def __init__(self, dim, dim_head, heads=4, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.drop = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            # nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)           # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)          # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        attn = self.drop(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
        # return out


class ConvLSTM(nn.Module):

    def __init__(self, input_size, num_channels, input_dim, hidden_dim, kernel_size, num_layers, dropout,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        # self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        for i in range(num_layers):
            hidden_dim[i] = hidden_dim[num_layers - 1]  # int(hidden_dim[num_layers-1]/2)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.tcn = TemporalConvNet(input_size, num_channels, hidden_dim[0], kernel_size[0], dropout)
        self.tcn = []
        self.att = []
        # for row in range(num_layers):
        #     self.tcn.append([])
        #     self.att.append([])
        # for row in range(num_layers):
        self.tcn = TemporalConvNet(input_size, num_channels, hidden_dim[0], kernel_size[0], dropout).to(self.device)
        self.att = Attention(hidden_dim[0], int(hidden_dim[0]/2)).to(self.device)

        self.dropout = nn.Dropout(0.2)

        # self.W_a = Parameter(torch.Tensor(self.hidden_dim[0], self.hidden_dim[0])).to(self.device)
        # self.U_a = Parameter(torch.Tensor(self.hidden_dim[0], self.hidden_dim[0])).to(self.device)
        # self.V_a = Parameter(torch.Tensor(self.hidden_dim[0], self.hidden_dim[0])).to(self.device) # For more than 2 lstm layers
        # self.V_a1 = Parameter(torch.Tensor(self.hidden_dim[0], self.input_dim)).to(self.device) # for 1 lstm layers
        # self.v_a = Parameter(torch.Tensor(1, self.hidden_dim[0])).to(self.device)

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        self.LN = nn.LayerNorm(hidden_dim[0], eps=0, elementwise_affine=True)
        self.LN1 = nn.BatchNorm1d(hidden_dim[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.reset_weigths()

    # def reset_weigths(self):
        """reset weights
        """
        # for weight in self.parameters():
        #     nn.init.xavier_normal_(weight)
        # nn.init.xavier_normal_(self.W_a)
        # nn.init.xavier_normal_(self.U_a)
        # nn.init.xavier_normal_(self.V_a)
        # nn.init.uniform_(self.v_a, -1, 1)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # input to LSTM: [seq_len, batch_size, input_size]
            # (t, b, c, h, w) -> (b, t, c, h, w)
            # input_tensor = input_tensor.permute(1, 0, 2)  #for fashionmnist
            input_tensor = input_tensor.permute(3, 0, 1, 2).squeeze()  #for cifar10

        _, b, _ = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b)

        # layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(0)
        cur_layer_input = input_tensor

        total_u = []
        stepsinsplit = []
        output_inner = []
        output4TCN = []
        for row in range(self.num_layers):
            output_inner.append([])

        for row in range(self.num_layers):
            output4TCN.append([])
            total_u.append([])
            stepsinsplit.append([])

        h = []
        for row in range(self.num_layers):
            h.append([])
        c = []
        for row in range(self.num_layers):
            c.append([])
        delta_u = []
        delta_u_bl = []
        gate = []
        for row in range(self.num_layers):
            delta_u.append([])
            delta_u_bl.append([])
            gate.append([])

        tcn_output = []  # 璁板綍tcn鍦ㄦ柇鐐圭殑杈撳嚭锛岀粰attention浣跨敤
        cut_timestep = []  # 璁板綍姣忎釜batch鐨勬柇鐐规椂闂存浣嶇疆
        once = []
        previous_t = []
        h_slot = []
        for row in range(self.num_layers):
            once.append([])
            previous_t.append([])
            tcn_output.append([])
            cut_timestep.append([])
            h_slot.append([])

        for layer_idx in range(self.num_layers):  # 鍒濆鍖?
            h[layer_idx], c[layer_idx] = hidden_state[layer_idx]
            once[layer_idx] = 0
            total_u[layer_idx] = torch.zeros([b, 1], dtype=torch.float).to(self.device)
            stepsinsplit[layer_idx] = 0
            h_slot[layer_idx] = h[layer_idx]

        for t in range(seq_len):
            # print(t)
            for layer_idx in range(self.num_layers):

                if layer_idx == 0:
                    cur_layer_input = input_tensor
                else:
                    cur_layer_input = self.dropout(cur_layer_input)  # dropout on each lstm layer

                h[layer_idx], delta_u[layer_idx] = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[t, :, :],
                    cur_state=h[layer_idx])

                delta_u_bl[layer_idx] = torch.mean(delta_u[layer_idx], dim=0)
                total_u[layer_idx] = total_u[layer_idx] + delta_u_bl[layer_idx]
                stepsinsplit[layer_idx] = stepsinsplit[layer_idx] + 1
                # U = torch.Tensor(1, 1)
                # U.data.uniform_(0, 1)
                # tau = 0.001
                # beta = inverse4sigmoid(2 * torch.sigmoid(torch.tensor(total_u[layer_idx])) - 1)
                # pointer = torch.sigmoid((beta + torch.log(U) - torch.log(1 - U)) / tau)
                pointer = torch.mean(total_u[layer_idx], dim=0)
                # total_u[layer_idx] = delta_u[layer_idx]

                # if torch.equal(h[layer_idx], h_slot[layer_idx]):
                #     output_inner[layer_idx].append(h[layer_idx])
                # else:
                #     output_inner[layer_idx].append(h[layer_idx] + h_slot[layer_idx])
                output_inner[layer_idx].append(h[layer_idx])
                current_output = torch.stack(output_inner[layer_idx], dim=1)

                # if layer_idx == self.num_layers - 1:
                if t == 0:
                    x = current_output.squeeze().unsqueeze(2)
                else:
                    x = current_output.permute(0, 2, 1)  # .cuda() .squeeze()

                if pointer >= 0.2:   # or t == int(seq_len/2) int(seq_len/2) or t == int(seq_len)

                    cut_timestep[layer_idx].append(t)
                    # print('stepsinsplit=', stepsinsplit[layer_idx])

                    # if layer_idx == self.num_layers - 1:  # ding ceng
                    if once[layer_idx] == 0:
                        previous_t[layer_idx] = t
                        x = self.dropout(x)
                        y = self.tcn(x)
                        # y1 = torch.tanh(y[:, :, -1])   # .unsqueeze(2)
                        y1 = self.LN(y[:, :, -1])
                        tcn_output[layer_idx].append(y1)
                        once[layer_idx] = 1
                    else:
                        x_seg = self.dropout(x[:, :, previous_t[layer_idx]:t])
                        y = self.tcn(x_seg)
                        # y1 = torch.tanh(y[:, :, -1])
                        y1 = self.LN(y[:, :, -1])
                        tcn_output[layer_idx].append(y1)
                        previous_t[layer_idx] = t

                    tcn_current_output = torch.stack(tcn_output[layer_idx], dim=1)

                    # selfatt = self.att(tcn_current_output) + tcn_current_output
                    selfatt = tcn_current_output
                    att_y_sum = selfatt[:, -1, :]

                    coeff = total_u[layer_idx] / stepsinsplit[layer_idx]
                    # tors = torch.cat((torch.exp(coeff), torch.exp(1 - coeff)), 1) + h_slot[layer_idx]
                    tors = torch.cat((coeff, 1 - coeff), 1)
                    coeff = F.softmax(tors, dim=1)
                    h[layer_idx] = coeff[:, 0].unsqueeze(1) * (att_y_sum) + \
                                   coeff[:, 1].unsqueeze(1) * (h[layer_idx])
                    # h[layer_idx] = gate[layer_idx] * (att_y_sum) + \
                    #                (1-gate[layer_idx]) * (h[layer_idx] )
                    #
                    # h[layer_idx] = gate[layer_idx] * h_slot[layer_idx] + \
                    #                coeff[:, 0].unsqueeze(1) * (att_y_sum) + \
                    #                coeff[:, 1].unsqueeze(1) * (h[layer_idx])

                    total_u[layer_idx] = torch.zeros([b, 1], dtype=torch.float).to(self.device)
                    stepsinsplit[layer_idx] = 0

                    h_slot[layer_idx] = h[layer_idx]

                # if layer_idx < self.num_layers - 1:
                #     h[layer_idx] = torch.rand(h[layer_idx].shape[0], h[layer_idx].shape[1]) * 1e-15

                if layer_idx < self.num_layers - 1:
                    # layer_output = torch.stack(output_inner[layer_idx], dim=1)
                    # temp_x = layer_output.permute(0, 2, 1)
                    # cur_layer_input = self.tcn[layer_idx](temp_x).permute(2, 0, 1)
                    # temp_x = current_output.permute(0, 2, 1)
                    # cur_layer_input = self.tcn[layer_idx](temp_x).permute(2, 0, 1)

                    cur_layer_input = current_output.permute(1, 0, 2)

                    # layer_output = torch.stack(output_inner[layer_idx], dim=1)
                    # cur_layer_input = layer_output.permute(1, 0, 2)

        # 最后对全部长度做卷积
        # x = self.dropout(x)
        y = self.tcn(x)
        y1 = self.LN1(y[:, :, -1])
        tcn_output[layer_idx].append(y1)
        tcn_current_output = torch.stack(tcn_output[layer_idx], dim=1)

        return tcn_current_output, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

