import os
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
        self.locations_vocab = set(df['source'])  # set of all locations
        self.vocab_size = len(self.locations_vocab)
        self.locations_to_ix = {l: i for i, l in enumerate(self.locations_vocab)}  # mapping every location to an index
        self.ix_to_locations = {i: l for i, l in self.locations_to_ix.items()}  # reverse the dict
        # Substitute the original location values with their index
        df['source'] = df['source'].map(self.locations_to_ix)
        df['target'] = df['target'].map(self.locations_to_ix)
        # for each location compute the probability (n occurrences / sum) and power it to 3/4
        probabilities = np.power(df.groupby('source').count() / df.groupby('source').count().sum(), 3 / 4)
        probabilities = probabilities.reset_index()
        probabilities.columns = ['location', 'prob']
        assert len(probabilities) == self.vocab_size
        # for each couple in the dataset sample one negative location, different from context and target
        # pick at random the negative samples. Some will be wrong, fixed later
        neg_data = probabilities.sample(
            n=len(df),
            weights='prob',
            replace=True,
            random_state=RANDOM_STATE
        )['location'].values
        df['negative'] = neg_data
        # select the wrong rows
        eq_mask = (df['source'] == df['negative']) | (df['target'] == df['negative'])
        cnt = np.sum(eq_mask)  # number of wrong sampling
        while np.sum(eq_mask) != 0:
            # while the number of wrong rows is different from zero, resample only the wrong rows
            df.loc[eq_mask, 'negative'] = probabilities.sample(
                n=cnt,
                weights='prob',
                replace=True,
            )['location'].values
            eq_mask = (df['source'] == df['negative']) | (df['target'] == df['negative'])
            cnt = np.sum(eq_mask)  # number of wrong sampling

        assert np.all(df['source'] != df['negative'])  # check if every source is different from its negative sample
        assert np.all(df['target'] != df['negative'])  # check if every target is different from its negative sample
        data = df.values
        # data is a tensor of (v, u, n) tuples where:
        # v = context , u = target , n = negative sampling
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


def train_model(epochs=15):
    os.chdir(r"G:\Download\computer science\datasets\Google Local\\")
    # The classification task is a fake task. We're interested in learning the embeddings of the locations.
    # We train the neural net on the whole dataset with the task of predicting the next location given the previous.
    # The layers are Embedding layers, which will learn the new projection of the original data into two spaces,
    # target and context respectively. We use negative sampling with a third random index for each couple.
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
    losses = []

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
        losses.append(average_loss)
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
    losses = pd.DataFrame({'loss': losses})
    losses.to_csv(f'./losses/losses_time_{round(time())}.csv', index=False)


def plot_loss():
    os.chdir(r"G:\Download\computer science\datasets\Google Local\losses\\")
    losses_file = os.listdir()[-1]
    losses = pd.read_csv(losses_file)
    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(
        losses.index + 1,
        losses['loss']
    )
    ax.set(xlabel='epoch', ylabel='loss', title='Loss over epochs')
    plt.show()


if __name__ == '__main__':
    train_model()
    # plot_loss()
