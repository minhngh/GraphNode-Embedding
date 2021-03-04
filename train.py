from tqdm import tqdm
from loss import BipartiteEdgePred
import torch
def train(model, features, adj, edges, degrees, args):
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    n_iters = 100
    criterion = BipartiteEdgePred()
    
    batch_size = args.batch_size
    n_iters = edges.shape[0] // batch_size
    for epoch in range(args.epochs):
        model.train()
        bar = tqdm(range(n_iters))
        for i in bar:
            optimizer.zero_grad()
            
            batch_edges = torch.LongTensor(edges[i * batch_size : (i + 1) * batch_size])

            embeddings = model(features, adj)

            loss = criterion(batch_edges[:, 0], batch_edges[:, 1], embeddings, degrees)
            loss.backward()
            optimizer.step()
            bar.set_description(f'ep:{epoch + 1} - iter:{i + 1} - loss: {loss.detach().item(): .4f}')
    
    return model