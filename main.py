import os
import pandas as pd
import torch
from torch.nn import Linear, Embedding, Module, NLLLoss
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from time import time
torch.manual_seed(883125)


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
        # the user is not used for the learning task
        # self.users_vocab = {}
        # self.users_to_ix = {}
        # self.ix_to_users = {}
        # Substitute the original location values with their index
        df['source'] = df['source'].map(self.locations_to_ix)
        df['target'] = df['target'].map(self.locations_to_ix)
        self.data = torch.LongTensor(df.values).to(_device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ixs):
        x, y = self.data[ixs, 0], self.data[ixs, 1]
        return x, y


class EmbeddingNet(Module):
    def __init__(self, vocabulary_size, embedding_dim=128, linear_dim=128):
        super(EmbeddingNet, self).__init__()
        self.embeddings = Embedding(vocabulary_size, embedding_dim)
        self.linear1 = Linear(embedding_dim, linear_dim)
        self.linear2 = Linear(linear_dim, vocabulary_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def train_model(resume=True):
    os.chdir(r"G:\Download\computer science\datasets\Google Local\\")
    # The classification task is a fake task. We're interested in learning the embeddings of the locations.
    # We train the neural net on the whole dataset with the task of predicting the next location given the previous.
    # The first layer is a Embedding layer, which will learn the new projection of the original data.
    # The net need be shallow, in order to capture the maximum amount of information in the embeddings.
    # Ultimately, we hope to discover some cluster pattern in the embeddings.

    # TODO: count average reviews per location
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    dataset = GoogleLocalDataset(device)
    dataloader = DataLoader(dataset, batch_size=256)
    model = EmbeddingNet(dataset.vocab_size).to(device)
    last_epoch = 0
    # resume from last checkpoint
    if resume:
        path = 'checkpoints'
        list_models = os.listdir(os.path.join(path))
        path += '/' + list_models[-1]
        last_epoch = int(path.split('_')[1])
        model.load_state_dict(torch.load(path))
    loss_function = NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = range(last_epoch+1, last_epoch+10)               # train 10 epochs at a time

    # TRAINING
    loss, losses = 0, []
    for epoch in epochs:
        total_loss = 0
        for context, target in tqdm(dataloader):
            model.zero_grad()
            log_probs = model(context)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # save network
        torch.save(
            model.state_dict(),
            f'./checkpoints/net_{epoch}_{round(time())}.pth'
        )
        print(f"\nend epoch: {epoch}\tcurrent loss: {loss}")
        losses.append(total_loss)
    print(losses)  # The loss should decrease every iteration over the training data!


if __name__ == '__main__':
    train_model()
