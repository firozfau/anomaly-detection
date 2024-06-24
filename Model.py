import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.functional import softplus
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
from torch.autograd import Variable
from Graph import returnA
from utils import GCN_Loss
import threading
import time

class GraphConvolution(Module):


    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        adj = adj.clone().detach()
        support = support.clone().detach()
        output = torch.spmm(adj.double(), support.double())
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, out):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, out)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        return x


class att_encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self,
                 input_size,
                 encoder_num_hidden,
                 batch_size,
                 window_length,
                 parallel=False):
        """Initialize an encoder in DA_RNN."""
        super(att_encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.batch_size = batch_size
        self.window_length = window_length
        self.parallel = parallel

        self.h_n = torch.nn.Parameter(torch.FloatTensor(1, batch_size, encoder_num_hidden), requires_grad=False)
        self.s_n = torch.nn.Parameter(torch.FloatTensor(1, batch_size, encoder_num_hidden), requires_grad=False)

        torch.nn.init.uniform_(self.h_n, a=0, b=0)
        torch.nn.init.uniform_(self.s_n, a=0, b=0)

        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )

        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.window_length,
            out_features=1
        )

    def forward(self, X):
        """forward.

        Args:
            X: input data

        """

        X_tilde = Variable(X.data.new(
            X.size(0), self.window_length, self.input_size).zero_())

        x = torch.cat((self.h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                       self.s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                       X.permute(0, 2, 1)), dim=2)

        x = self.encoder_attn(
            x.view(-1, self.encoder_num_hidden * 2 + self.window_length))

        alpha = F.softmax(x.view(-1, self.input_size), dim=1)

        for t in range(self.window_length):
            x_tilde = torch.mul(alpha, X[:, t, :])
            X_tilde[:, t, :] = x_tilde

        self.encoder_lstm.flatten_parameters()

        _, final_state = self.encoder_lstm(x_tilde.unsqueeze(0), (self.h_n, self.s_n))
        self.h_n = Parameter(final_state[0])
        self.s_n = Parameter(final_state[1])
        X_tilde = X_tilde.view(-1, self.window_length * self.input_size)

        return X_tilde


def tabular_encoder(input_size: int, latent_size: int):
    return nn.Sequential(
        nn.Linear(input_size, latent_size * 2),
    )


def tabular_decoder(latent_size: int, output_size: int):
    return nn.Sequential(
        nn.Linear(latent_size, output_size * 2),
    )


class MUTANT(nn.Module):

    def __init__(self, input_size: int, w_size, hidden_size: int, latent_size, batch_size, window_length, out_dim):
        super().__init__()
        self.input_size = input_size
        self.w_size = w_size
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.att_encoder = att_encoder(input_size, hidden_size, batch_size, out_dim)
        self.encoder = tabular_encoder(w_size, latent_size)
        self.decoder = tabular_decoder(latent_size, w_size)
        self.GCN = GCN(window_length, out_dim)
        self.prior = Normal(0, 1)
        self.stop_loading = False

    def forward(self, x):
        Xt = []
        l0 = 0
        for i in x:
            A = torch.tensor(returnA(i))
            x_g = self.GCN(i.permute(1, 0), A).permute(1, 0)
            loss = GCN_Loss(x_g)
            l0 = loss
            x_g = x_g.detach().numpy()
            Xt.append(x_g)

        Xt = np.array(Xt)  # Convert list of numpy arrays to a single numpy array
        Xt = torch.tensor(Xt, dtype=torch.float32)  # Convert numpy array to tensor

        X_w = self.att_encoder(Xt)
        pred_result = self.predict(X_w)
        mu = pred_result['recon_mu']
        Xt = Xt.view(-1, Xt.shape[1] * Xt.shape[2])
        loss_function = torch.nn.MSELoss()
        l1 = loss_function(mu, Xt)
        l2 = torch.mean(-0.5 * torch.sum(1 + pred_result['latent_sigma'] - pred_result['latent_mu'].pow(2) - pred_result['latent_sigma'].exp(), dim=1), dim=0)
        loss = l0 + l1 + l2
        return loss

    def predict(self, x) -> dict:
        batch_size = len(x)
        latent_mu, latent_sigma = self.encoder(x).chunk(2, dim=1)
        latent_sigma = softplus(latent_sigma) + 1e-4
        dist = Normal(latent_mu, latent_sigma)
        z = dist.rsample()
        z = z.view(batch_size, self.latent_size)
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma) + 1e-4
        recon_mu = recon_mu.view(-1, x.shape[1])
        recon_sigma = recon_sigma.view(-1, x.shape[1])

        return dict(latent_dist=dist, latent_mu=latent_mu,
                    latent_sigma=latent_sigma, recon_mu=recon_mu,
                    recon_sigma=recon_sigma, z=z)

    def loader_reconstructed(self):
        # Placeholder for actual implementation
        time.sleep(0.1)  # Simulate some processing time
        return np.random.random(10)  # Return some dummy data
    def loader(self):
        while not self.stop_loading:
            for cursor in '|/-\\':
                if self.stop_loading:
                    break
                print(f'\r Data-calculate- {cursor}', end='', flush=True)
                time.sleep(0.1)

    def is_anomaly(self, x, num_t, con_t):
        score = []
        np.set_printoptions(threshold=999999999999999999)

        self.stop_loading = False  # Ensure stop_loading is reset
        loader_thread = threading.Thread(target=self.loader)
        loader_thread.start()


        try:
            for i, inputs in enumerate(x, 0):
                p = self.reconstructed_probability(inputs)

                if i == num_t:
                    p = p[:con_t]
                score = np.concatenate((score, p), axis=0)

        finally:
            self.stop_loading = True
            loader_thread.join()

        score = np.array(score)
        print("-Done\n")
        return score



    def reconstructed_probability(self, x):

        with torch.no_grad():
            Xt = []
            for i in x:

                A = torch.tensor(returnA(i))

                x_g = self.GCN(i.permute(1, 0), A).permute(1, 0).detach().numpy()
                Xt.append(x_g)


            Xt = np.array(Xt)  # Convert list of numpy arrays to a single numpy array
            Xt = torch.tensor(Xt, dtype=torch.float32)  # Convert numpy array to tensor
            X_w = self.att_encoder(Xt)
            pred = self.predict(X_w)
            mu = pred['recon_mu']
            Xt = Xt.view(-1, Xt.shape[1] * Xt.shape[2])

            p = []
            for i in range(x.shape[0]):
                t = abs(torch.sum((Xt[i] - mu[i])))
                p.append(t)
        p = np.array(p)
        return p

    def generate(self, batch_size: int = 1) -> torch.Tensor:
        z = self.prior.sample((batch_size, self.latent_size))
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        return recon_mu + recon_sigma * torch.rand_like(recon_sigma)
