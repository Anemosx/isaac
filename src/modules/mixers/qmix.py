import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        # agent_qs [32, 200, 8]
        # states [32, 200, 201, 8, 89]
        # self.state_dim 143112
        bs = agent_qs.size(0)
        # bs = 32
        states = states.reshape(-1, self.state_dim)
        # states [6400, 143112]
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # agent_qs [6400, 1, 8]
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        # w1 [6400, 256]
        b1 = self.hyper_b_1(states)
        # b1 [6400, 32]
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        # w1 [6400, 8, 32]
        b1 = b1.view(-1, 1, self.embed_dim)
        # b1 [6400, 1, 32]

        # agent_qs [6400, 1, 8]
        # w1 [6400, 8, 32]
        # b1 [6400, 1, 32]
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # hidden [6400, 1, 32]

        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        # w_final [6400, 32]
        w_final = w_final.view(-1, self.embed_dim, 1)
        # w_final [6400, 32, 1]
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # v [6400, 1, 1]

        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # y [6400, 1, 1]
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        # q_tot [32, 200, 1]
        return q_tot

class ConcatStateHistoryQMixer(QMixer):
    def __init__(self, args):
        self.state_dim = int(np.prod(args.state_shape))
        self.history_dim = int(np.prod(args.history_dim))
        self.state_dim += self.history_dim
        args.state_shape = self.state_dim
        super(ConcatStateHistoryQMixer, self).__init__(args)

    def forward(self, agent_qs, states, histories):
        bs = agent_qs.size(0)
        states = th.cat([states.view(bs, -1), histories.view(bs, -1)], dim=1)
        return super(ConcatStateHistoryQMixer, self).forward(agent_qs, states)

class UnconstrainedQMixer(QMixer):
    def __init__(self, args):
        super(UnconstrainedQMixer, self).__init__(args)
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.agent_input_size = int(args.n_agents*args.n_actions)
        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.agent_input_size)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.agent_input_size))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.agent_input_size)
        # First layer
        w1 = self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.agent_input_size, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = self.hyper_w_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

class DoubleQMixer(nn.Module):
    def __init__(self, args):
        super(DoubleQMixer, self).__init__()
        self.unconstrained_mixer = UnconstrainedQMixer(args)
        self.state_dim = np.prod(args.state_shape)
        self.n_agents = args.n_agents
        self.unconstrained_input_size = self.state_dim + self.n_agents*args.n_actions
        args.state_shape = [args.rnn_hidden_dim, args.n_agents]
        self.constrained_mixer = QMixer(args)

    def forward(self, actions, agent_qs, states, hidden_states):
        bs = agent_qs.size(0)
        states = states.reshape(bs, -1, self.state_dim)
        agent_qs = agent_qs.view(bs, -1, self.n_agents)
        return self.unconstrained_mixer(actions, states), self.constrained_mixer(agent_qs, hidden_states)