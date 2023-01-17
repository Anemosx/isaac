import torch.nn as nn
import torch as th


class Discriminator(nn.Module):
    # setup discriminator for behavior imitation
    def __init__(self, args, history_dim, action_dim):
        super(Discriminator, self).__init__()

        self.args = args
        self.n_agents = args.n_agents

        hypernet_embed = 256
        hypernet_embed1 = 64
        hypernet_embed_action = 32

        self.hyper_w_1_test = nn.Sequential(nn.Linear(history_dim, hypernet_embed * 2),
                                            nn.LeakyReLU(0.25),
                                            nn.Linear(hypernet_embed * 2, hypernet_embed),
                                            nn.LeakyReLU(0.25),
                                            nn.Linear(hypernet_embed, hypernet_embed))

        self.hyper_a_1_test = nn.Sequential(nn.Linear(action_dim, hypernet_embed1),
                                            nn.ReLU(),
                                            nn.Linear(hypernet_embed1, hypernet_embed_action))

        self.hyper_a_big_test = nn.Sequential(nn.Linear(hypernet_embed_action + hypernet_embed, hypernet_embed),
                                              nn.LeakyReLU(0.25),
                                              nn.Linear(hypernet_embed, hypernet_embed),
                                              nn.LeakyReLU(0.25),
                                              nn.Linear(hypernet_embed, hypernet_embed1),
                                              nn.LeakyReLU(0.25),
                                              nn.Linear(hypernet_embed1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, agent_actions, states):
        # forward pass with action and history inputs
        bs = agent_actions.size(0)
        epi_length = agent_actions.size(1)
        states = states.swapaxes(2, 3)
        states = states.reshape(bs * epi_length, self.n_agents, -1)
        agent_actions = agent_actions.reshape(bs * epi_length, self.n_agents, -1)

        agent_actions = self.hyper_a_1_test(agent_actions)
        w1 = th.abs(self.hyper_w_1_test(states))

        input_1 = th.cat([agent_actions, w1], dim=2)
        out = self.hyper_a_big_test(input_1)
        output = self.sigmoid(out)

        prediction = output.view(bs, -1, self.n_agents, 1)

        return prediction


class DiscriminatorState(nn.Module):
    # setup classifier for mimic feature states
    def __init__(self, args, history_dim):
        super(DiscriminatorState, self).__init__()

        self.args = args
        self.n_agents = args.n_agents

        hypernet_embed = 256
        hypernet_embed1 = 64

        self.hyper = nn.Sequential(nn.Linear(history_dim, hypernet_embed * 4),
                                   nn.LeakyReLU(0.25),
                                   nn.Linear(hypernet_embed * 4, hypernet_embed * 2),
                                   nn.LeakyReLU(0.25),
                                   nn.Linear(hypernet_embed * 2, hypernet_embed),
                                   nn.LeakyReLU(0.25),
                                   nn.Linear(hypernet_embed, hypernet_embed1),
                                   nn.LeakyReLU(0.25),
                                   nn.Linear(hypernet_embed1, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, states):
        # forward pass of state features
        bs = states.size(0)
        epi_length = states.size(1)
        states = states.reshape(bs * epi_length, -1)

        out = self.hyper(states)
        output = self.sigmoid(out)

        prediction = output.view(bs, epi_length, 1)

        return prediction
