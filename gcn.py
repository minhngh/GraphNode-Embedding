import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weight(modules, activation):
    for m in modules:
        if isinstance(m, nn.Linear):
            if activation:
                nn.init.xavier_uniform_(m.weight.data, gain = nn.init.calculate_gain(activation))
            else:
                nn.init.xavier_uniform_(m.weight.data)
            if m.bias:
                nn.init.constant_(m.bias.data, 0.)

def get_activation_function(activation):
    if activation == 'sigmoid': return nn.Sigmoid()
    elif activation == 'relu': return nn.ReLU()
    elif activation == 'tanh': return nn.Tanh()

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super().__init__()
        self.activation = get_activation_function(activation) if activation else None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim, bias = False)
        init_weight(self.modules(), activation)
    def forward(self, input, A_hat):
        output = self.fc(input)
        output = torch.matmul(A_hat, output)
        if self.activation:
            output = self.activation(output)
        return F.normalize(output)
class GCNNet(nn.Module):
    def __init__(self, num_layers, in_dim, out_dim, activation):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.layers = nn.ModuleList([GCN(in_dim, out_dim, activation)])
        for _ in range(num_layers - 1):
            self.layers.append(GCN(out_dim, out_dim, activation))
        self.fc = nn.Linear(in_dim + out_dim * num_layers, out_dim, bias = False)
        init_weight(self.modules(), activation)
    
    def forward(self, input, A_hat):
        concats = [input]
        output = input
        for layer in self.layers:
            output = layer(output, A_hat)
            concats.append(output)
        concats = torch.cat(concats, dim = 1)
        output = self.fc(concats)
        return F.normalize(output)

