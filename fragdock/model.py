import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, num_hidden_layer=2,
                 output_dim=512, bias=True):
        super(FCN, self).__init__()

        layer = list()
        for i in range(0, num_hidden_layer):
            ini = hidden_dim
            fin = hidden_dim

            if i == 0:
                ini = input_dim
            w_tmp = nn.Linear(ini, fin, bias=bias)
            nn.init.xavier_normal_(w_tmp.weight.data)
            if w_tmp.bias is not None:
                nn.init.normal_(w_tmp.bias.data)

            layer.append(w_tmp)
            layer.append(nn.ReLU())

        w_tmp = nn.Linear(hidden_dim, output_dim, bias=bias)
        nn.init.xavier_normal_(w_tmp.weight.data)
        if w_tmp.bias is not None:
            nn.init.normal_(w_tmp.bias.data)
        layer.append(w_tmp)
        self.layer = nn.Sequential(*layer)

    def forward(self, X):

        Y = self.layer(X)

        return Y


class Net(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim1=1024, num_hidden_layer1=2,
                 latent_dim=512, hidden_dim2=1024, num_hidden_layer2=2,
                 output_dim=1, bias=True):
        super(Net, self).__init__()

        self.fc_i = FCN(input_dim=input_dim, hidden_dim=hidden_dim1,
                        num_hidden_layer=num_hidden_layer1,
                        output_dim=latent_dim, bias=bias)
        self.fc_o = FCN(input_dim=latent_dim*2, hidden_dim=hidden_dim2,
                        num_hidden_layer=num_hidden_layer2,
                        output_dim=output_dim, bias=bias)

    def fo_i(self, x):
        z = self.fc_i(x)
        return z

    def fo_f(self, z1, z2):
        z = torch.cat([z1, z2], dim=1)

        z = torch.relu(z)
        y = self.fc_o(z)
        return y

    def forward(self, x1, x2):
        z1 = self.fo_i(x1)
        z2 = self.fo_i(x2)
        y = self.fo_f(z1, z2)

        return y
