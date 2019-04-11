import torch.nn as nn
import torch


class Generator(nn.Module):

    def __init__(self, nin, nout):
        super(Generator, self).__init__()
        nf = 12
        self.net = nn.Sequential(

            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=nin, out_channels=nf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nout),
            nn.ReLU(inplace=True),

            # state size. (nf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=nf * 8, out_channels=nf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(inplace=True),

            # state size. (nf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(inplace=True),

            # state size. (nf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(16, 8, 4),
            nn.AvgPool2d(16, 8, 4),
            nn.AvgPool2d(16, 8, 4),
            nn.AvgPool2d(2, 2, 1),
            # state size. (nf) x 32 x 32
            nn.ConvTranspose2d(in_channels=nf, out_channels=nout, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=nout,
            hidden_size=nout,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

    def forward(self, x):
        # print("-G-")
        # print(x.shape)
        seq = torch.reshape(x, (x.shape[1], 1, 1, x.shape[2]))
        # print(seq.shape)
        # print("...")
        out = self.net(seq)
        # print(out.shape)
        # print("...")
        out = torch.reshape(out, (out.shape[0], out.shape[2] * out.shape[3], out.shape[1]))
        # print(out.shape)
        # print("...")
        out, (h_n, h_c) = self.rnn(out)
        # print(out.shape)
        # print(h_n.shape)
        # print(h_c.shape)
        # print("-G-")
        return h_c


class Descriminator(nn.Module):

    def __init__(self, nc, nf):
        super(Descriminator, self).__init__()
        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=nf,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Sequential(
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        # x = torch.reshape(x,(x.shape[1],1,1,x.shape[2]))
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out
