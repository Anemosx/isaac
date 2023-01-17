import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .pfrnn import PFGRUCell

class PFRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(PFRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = PFGRUCell(args.num_particles, args.rnn_hidden_dim, args.rnn_hidden_dim, args.ext_obs, args.ext_act, args.resamp_alpha)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.num_particles = self.rnn.num_particles

    def init_hidden(self, batch_size, n_agents):
        new_batch_size = batch_size*n_agents
        h0 = th.rand(new_batch_size * self.num_particles, self.args.rnn_hidden_dim)
        p0 = th.ones(new_batch_size * self.num_particles, 1) * np.log(1 / self.num_particles)
        return (h0, p0)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        x = x.repeat(self.num_particles, 1)
        h0, p0 = hidden_state
        h_in = h0.reshape(-1, self.args.rnn_hidden_dim)
        p_in = p0.reshape(-1, 1)
        h1, p1 = self.rnn(x, (h_in, p_in))
        p1 = p1.view(self.num_particles, -1, 1)
        h1 = h1.view(self.num_particles, -1, self.args.rnn_hidden_dim)
        y = h1 * th.exp(p1)
        y = th.sum(y, dim=0)
        q = self.fc2(y)
        return q, (h1, p1)
