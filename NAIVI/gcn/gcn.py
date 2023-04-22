import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
from .utils import normalize


class GCNModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNModel, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GCN:

    def __init__(
            self,
            adjacency_matrix: torch.Tensor,
            features: torch.Tensor,
            labels: torch.Tensor,
            idx_train: torch.Tensor,
            idx_test: torch.Tensor,
            n_hidden: int = 16,
            dropout: float = 0.5,
    ):
        self.adjacency_matrix = normalize(adjacency_matrix + torch.eye(adjacency_matrix.shape[0]))
        self.features = normalize(features)
        self.labels = labels
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.model = GCNModel(
            nfeat=self.features.shape[1],
            nhid=self.n_hidden,
            nclass=self.labels.max().item() + 1,
            dropout=self.dropout,
        )


    def fit(
            self,
            max_iter: int = 200,
            learning_rate: float = 0.01,
            weight_decay: float = 5e-4,
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in range(max_iter):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(self.features, self.adjacency_matrix)
            loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
            loss_train.backward()
            optimizer.step()
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.item()))

    def output(self):
        logits = self.model(self.features, self.adjacency_matrix)
        probas = torch.softmax(logits, dim=1)
        return dict(
            pred_binary_covariates=torch.cat([self.features, probas], dim=1)
        )
