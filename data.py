import json
import numpy as np
import networkx as nx
import torch


def convert_labels_to_onehot(node2label):
    labels = np.array(list(node2label.values()))
    unique_labels = np.unique(labels)

    new_labels = dict(zip(unique_labels.tolist(), range(unique_labels.size)))
    node2newlabel = {node: new_labels[lb] for node, lb in node2label.items()}
    return node2newlabel

def get_normalized_adj(g):
    num_nodes = len(g.nodes)
    indexs1 = torch.LongTensor(np.array(list(g.edges)).T)
    indexs2 = torch.LongTensor(np.array(list(g.edges))[:, [1, 0]].T)
    indexs3 = torch.LongTensor(np.vstack((range(len(g.nodes)), range(len(g.nodes)))))
    indexs = torch.cat((indexs1, indexs2, indexs3), dim=1)
    values = []
    for i in range(indexs1.shape[1]):
        values.append(1/(np.sqrt(g.degree(int(indexs1[0][i])) + 1) * np.sqrt(g.degree(int(indexs1[1][i])) + 1))) 
    for i in range(indexs2.shape[1]):
        values.append(1/(np.sqrt(g.degree(int(indexs2[0][i])) + 1) * np.sqrt(g.degree(int(indexs2[1][i])) + 1))) 
    for i in range(indexs3.shape[1]):
        values.append(1/(np.sqrt(g.degree(int(indexs3[0][i])) + 1) * np.sqrt(g.degree(int(indexs3[1][i])) + 1))) 
    
    values = torch.FloatTensor(np.array(values))
    adj = torch.sparse.FloatTensor(indexs, values, torch.Size([num_nodes, num_nodes]))
    return adj
def read_graph(path):
    nodes = []
    edges = []
    node2label = {}
    with open(path, 'r') as f:
        for line in f:
            cols = line.split()
            if len(cols) == 0: continue
            if cols[0] == 't' and int(cols[-1]) != 0: break
            if cols[0] == 'v':
                nodes.append(int(cols[1]))
                node2label[int(cols[1])] = int(cols[2])
            elif cols[0] == 'e':
                edges.append((int(cols[1]), int(cols[2])))
    node2newlabel = convert_labels_to_onehot(node2label)

    with open('saved/label.json', 'w') as f:
        f.write(json.dumps(node2newlabel))

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    for node in g.nodes:
        # print( node2newlabel)
        g.nodes[node]['label'] = node2newlabel[node]
    return g

def get_data(g):
    num_classes = 0
    labels = []
    for node in g.nodes:
        labels.append(g.nodes[node]['label'])
        num_classes = max(num_classes, g.nodes[node]['label'])
    num_classes += 1
    features = np.eye(num_classes)[np.array(labels)]
    adj = get_normalized_adj(g)

    degrees = np.array([g.degree(node) for node in g.nodes])

    return torch.Tensor(features), adj, degrees, labels
