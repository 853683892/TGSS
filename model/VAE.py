import torch
import torch.nn.functional as F
from torch import nn

def cosine_similarity(p, z, average=True):
    p = F.normalize(p, p=2, dim=1)
    z = F.normalize(z, p=2, dim=1)
    loss = -(p * z).sum(dim=1)
    loss = loss.mean()
    return loss


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, emb_dim, loss, detach_target, beta=1):
        super(VariationalAutoEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.loss = loss
        self.detach_target = detach_target
        self.beta = beta

        self.criterion = cosine_similarity


        self.fc_mu = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc_var = nn.Linear(self.emb_dim, self.emb_dim)

        self.fc_decoder_1 = nn.Linear(self.emb_dim,self.emb_dim)
        self.m = nn.BatchNorm1d(self.emb_dim)
        self.fc_decoder_2 = nn.Linear(self.emb_dim,self.emb_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

        return



    def encode(self, x):
        'fc_mu 和 fc_var一样，用一个应该就可以'
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparamaterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        y = y.detach()

        mu, log_var = self.encode(x)

        z = self.reparamaterize(mu, log_var)

        y_hat = self.decoder(z)

        reconstruction_loss = self.criterion(y_hat, y)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = reconstruction_loss + self.beta * kl_loss

        return loss




























