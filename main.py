import os
from operator import neg

import pandas as pd
import numpy as np
import torch
from torch.nn import Embedding, Module
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

RANDOM_STATE = 883125
torch.manual_seed(RANDOM_STATE)


class GoogleLocalDataset(Dataset):
    def __init__(self, _device):
        df = pd.read_csv(
            "covisits.csv",
            usecols=['source', 'target']
        )
        # The query is a cartesian product, the resulting matrix is symmetric.
        # Every location appears at least one time in the target column
        self.locations_vocab = set(df['source'])                                    # set of all locations
        self.vocab_size = len(self.locations_vocab)
        self.locations_to_ix = {l: i for i, l in enumerate(self.locations_vocab)}   # mapping every location to an index
        self.ix_to_locations = {i: l for i, l in self.locations_to_ix.items()}      # reverse the dict
        # Substitute the original location values with their index
        df['source'] = df['source'].map(self.locations_to_ix)
        df['target'] = df['target'].map(self.locations_to_ix)
        # for each location compute the probability (n occurrences / sum) and power it to 3/4
        probabilities = np.power(df.groupby('source').count() / df.groupby('source').count().sum(), 3 / 4)
        probabilities = probabilities.reset_index()
        probabilities.columns = ['location', 'prob']
        assert len(probabilities) == self.vocab_size
        # for each couple in the dataset sample one negative location
        # it might be that the negative location appears in the context, more sophisticated implementation needed
        neg_data = probabilities.sample(
            n=len(df['source']),
            replace=True,
            weights='prob',
            random_state=RANDOM_STATE
        )['location']
        assert len(neg_data) == len(df)
        data = np.column_stack((df.values, neg_data.values.T))
        # data is a tensor of (v, u, n) tuples where:
        # u = target , v = context , n = negative sampling
        self.data = torch.LongTensor(data).to(_device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ixs):
        v, u, n = self.data[ixs, 0], self.data[ixs, 1], self.data[ixs, 2]
        return v, u, n


class EmbeddingNet(Module):
    def __init__(self, vocabulary_size, embedding_dim=256):
        super(EmbeddingNet, self).__init__()
        self.emb_size = vocabulary_size
        self.emb_dimension = embedding_dim
        # center embedding matrix
        self.u_embeddings = Embedding(vocabulary_size, embedding_dim, sparse=True)
        # neighbor embedding matrix
        self.v_embeddings = Embedding(vocabulary_size, embedding_dim, sparse=True)
        init_range = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-init_range, init_range)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        neg_emb_v = self.v_embeddings(neg_v)
        # positive score
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        # positive score
        neg_score = torch.mul(emb_u, neg_emb_v).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        return -1 * (torch.sum(score) + torch.sum(neg_score))


def train_model(epochs=10):
    os.chdir(r"G:\Download\computer science\datasets\Google Local\\")
    # The classification task is a fake task. We're interested in learning the embeddings of the locations.
    # We train the neural net on the whole dataset with the task of predicting the next location given the previous.
    # The first layer is a Embedding layer, which will learn the new projection of the original data.
    # The net need be shallow, in order to capture the maximum amount of information in the embeddings.
    # Ultimately, we hope to discover some cluster pattern in the embeddings.

    # check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # define dataset
    dataset = GoogleLocalDataset(device)
    # define dataloader
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # define model
    embedding_dim = 256
    model = EmbeddingNet(
        vocabulary_size=dataset.vocab_size,
        embedding_dim=256
    ).to(device)
    # define optimizer
    optimizer = optim.SparseAdam(model.parameters(), lr=1e-3)
    losses = pd.DataFrame(columns=['losses'])
    # resume from last checkpoint
    # if resume:
    #     path = 'checkpoints/'
    #     list_models = os.listdir(os.path.join(path[:-1]))
    #     if len(list_models):
    #         path += list_models[-1]
    #         checkpoint = torch.load(path)
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         epoch = checkpoint['epoch']

    # TRAINING
    model.train()
    epochs = range(epochs)
    for epoch in epochs:
        total_loss = 0
        process_bar = tqdm(dataloader)
        # u = target , v = context , n = negative sampling
        for u, v, n in process_bar:
            model.zero_grad()
            loss = model(u, v, n)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print current mini-batch loss
            process_bar.set_description("Loss: %0.8f" % loss.item())
        average_loss = total_loss / len(process_bar)
        losses.append({'loss': average_loss}, ignore_index=True)
        # save network
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'./checkpoints/trained_net_epoch_{epoch}_time_{round(time())}.pth')
        print(f"\nend epoch: {epoch}\tcurrent average loss: {average_loss}")
    # Save the DataFrames
    weights = model.u_embeddings.weight.detach().clone().cpu().numpy()
    embeddings = pd.DataFrame(weights, columns=[f"dim_{i}" for i in range(embedding_dim)])
    embeddings.to_csv(f'./embeddings/embeddings_time_{round(time())}.csv', index=False)
    losses.index.rename('index')
    losses.to_csv(f'./losses/losses_time_{round(time())}.csv')


def plot_loss():
    os.chdir(r"G:\Download\computer science\datasets\Google Local\losses\\")
    losses_file = os.listdir(os.path.join('checkpoints'))[-1]
    losses = pd.read_csv(losses_file)
    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(
        losses.index,
        losses['loss']
    )
    ax.set(xlabel='epoch', ylabel='loss', title='Loss over epochs')
    plt.show()


if __name__ == '__main__':
    train_model()
    plot_loss()
