import argparse
from data import read_graph, get_data
from gcn import GCNNet
from train import train
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default = 8, type = int)
    parser.add_argument('--learning-rate', default = 1e-3, type = float)
    parser.add_argument('--epochs', default = 50, type = int)
    parser.add_argument('--num-layers', default = 2, type = int)
    parser.add_argument('--out-dim', default = 20, type = int)
    parser.add_argument('--activation', default = 'tanh', type = str)
    parser.add_argument('--data-path', required = True, type = str)
    parser.add_argument('--cuda', default = True, type = bool)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    g = read_graph(args.data_path)

    features, adj, degrees, labels = get_data(g)
    edges = np.array(list(g.edges))
    
    features = features.to(device)
    adj = adj.to(device)

    model = GCNNet(args.num_layers, features.shape[1], args.out_dim, args.activation)
    model = train(model, features, adj, edges, degrees, args)

    torch.save(torch.LongTensor(labels), 'saved/labels.pt')
    torch.save(adj, 'saved/adj.pt')
    torch.save(features, 'saved/features.pt')
    torch.save(model, 'saved/model.pt')
    torch.save(edges, 'saved/edges.pt')