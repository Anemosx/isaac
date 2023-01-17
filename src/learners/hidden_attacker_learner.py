import numpy as np
from torch.optim import RMSprop
import torch as th
import torch.nn.functional as F
from components.episode_buffer import EpisodeBatch
from modules.agents.discriminator import Discriminator, DiscriminatorState


class HiddenAttackerLearner:
    # setup ISAAC Invader / Naive Invader
    def __init__(self, learner, prey_mac, scheme, args, env):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = learner.mac
        self.prey_mac = prey_mac
        self.learner = learner
        self.n_antagonists = self.args.n_antagonists
        self.n_actions = self.args.n_actions

        # get index for neighbourhood comparison (initially used but then abandoned)
        select_pos = [[ag_i * env.get_ally_num_attributes() + env.ally_state_attr_names.index('rel_x'),
                       ag_i * env.get_ally_num_attributes() + env.ally_state_attr_names.index('rel_y')]
                      for ag_i in range(self.n_agents)]
        self.select_pos = [coord for pos in select_pos for coord in pos]
        self.n_coords = 2

        # input dimension for discriminator
        self.mac_input_shape = int(np.prod(self.mac._get_input_shape(scheme)))
        self.history_dim_shape = self.mac_input_shape
        history_dim = (self.args.episode_limit + 1) * self.history_dim_shape

        # setup of discriminator
        self.discriminator = Discriminator(self.args, history_dim, self.args.n_actions)
        self.optimiser_dis = RMSprop(params=self.discriminator.parameters(), lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # bce for binary classification
        self.criterion = th.nn.BCELoss()
        self.counter_train = 0
        self.action_imit_interval = self.args.action_imit_interval

        # neighbourhood comparison (abandoned, but works well!)
        if self.args.step_neighborhood:
            self.move_distance = th.tensor(self.args.step_neighborhood_eps)

        # mimic state feature classifier
        if self.args.step_classifier:
            self.discriminator_states = DiscriminatorState(self.args, self.args.n_antagonists * (self.n_coords + 1))
            self.optimiser_dis_states = RMSprop(params=self.discriminator_states.parameters(), lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # pure imitation
        if self.args.only_gan:
            self.optimiser = RMSprop(params=self.learner.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train_gan(self, batch: EpisodeBatch):
        # train discriminator / behavior imitation
        history_batch, loss_dis, train_accuracy_tot, train_accuracy, train_accuracy_p = self.train_discriminator(batch)

        # get output from attacker
        action_dis, _ = self.mac_outs(self.mac, batch)

        # forward pass on discriminator
        predictions = self.discriminator(action_dis, history_batch)
        predictions = predictions[:, :, self.args.n_agents - self.args.n_antagonists:self.args.n_agents]
        predictions = th.where(predictions >= 0., predictions, th.zeros(1))
        predictions = th.where(predictions <= 1., predictions, th.ones(1))
        label_ant_p = th.ones(predictions.size())

        # behavior shaping
        if self.args.label_ign_r_states:
            rewards = batch["reward"][:, :-1]
            for epi_i in range(rewards.size(0)):
                for step_i in range(rewards.size(1)):
                    if th.abs(rewards[epi_i][step_i][0]) > self.args.ignore_threshold:
                        label_ant_p[epi_i][step_i] = predictions[epi_i][step_i]

        # loss for behavior imitation
        loss_ant = self.criterion(predictions, label_ant_p.detach())

        return loss_dis, loss_ant, train_accuracy_tot, train_accuracy, train_accuracy_p

    def train_discriminator(self, batch: EpisodeBatch):

        # discriminator

        # make history inputs and get outputs from protagonist and attacker
        history_batch = self.make_history_batch(batch)
        history_batch_p = history_batch
        action_dis, action_dis_p = self.mac_outs(self.mac, batch, self.prey_mac)

        # forward pass of discriminator with protagonist and attacker (action, histories)
        predictions_p = self.discriminator(action_dis_p, history_batch_p)
        predictions = self.discriminator(action_dis, history_batch)
        predictions_p = predictions_p[:, :, self.args.n_agents - self.args.n_antagonists:self.args.n_agents]
        predictions = predictions[:, :, self.args.n_agents - self.args.n_antagonists:self.args.n_agents]

        # BCE for loss
        label_p = th.ones(predictions_p.size())
        label_a = th.zeros(predictions.size())
        loss_p = self.criterion(predictions_p, label_p)
        loss_a = self.criterion(predictions, label_a)

        # training accuracy of discriminator
        train_accuracy_tot, train_accuracy, train_accuracy_p = calc_accuracy(predictions, predictions_p)

        # loss of discriminator
        loss_dis = loss_p + loss_a

        # backpropagation
        self.optimiser_dis.zero_grad()
        loss_dis.backward()
        th.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.args.grad_norm_clip)
        self.optimiser_dis.step()

        return history_batch, loss_dis, train_accuracy_tot, train_accuracy, train_accuracy_p

    def train(self, batch: EpisodeBatch, batch_p: EpisodeBatch, t_env: int, episode_num: int):
        # train ISAAC Invader / Naive Invader
        self.counter_train += 1
        loss_ant = th.tensor(0.)
        loss_dis, mac_loss, train_accuracy_tot, train_accuracy, train_accuracy_p, accuracy_step_tot, accuracy_step, accuracy_step_p = None, None, None, None, None, None, None, None

        # only imitation
        if self.args.only_gan:
            loss_dis, loss_ant, train_accuracy_tot, train_accuracy, train_accuracy_p = self.train_gan(batch)
            self.optimiser.zero_grad()
            loss_ant.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.learner.params, self.args.grad_norm_clip)
            self.optimiser.step()

        # Naive Invader
        elif self.args.obvious_antagonist:
            # only flip rewards and use normal algorithm
            mac_loss = self.learner.train(batch=batch, t_env=t_env, episode_num=episode_num, negative_reward=True)

        # ISAAC Invader
        else:
            # train behavior according to behavior imitation training scheduling
            if self.counter_train % self.action_imit_interval == 0:
                loss_dis, loss_ant, train_accuracy_tot, train_accuracy, train_accuracy_p = self.train_gan(batch)

            # mimic state features
            if self.args.step_imit:
                # use mimic state feature classifier
                if self.args.step_classifier:
                    # reward shaping according to mimic state feature classifier
                    rewards, accuracy_step_tot, accuracy_step, accuracy_step_p = self.classify_steps(batch, batch_p)
                else:
                    # neighbourhood comparison, measure epsilon distance to ISAAC Invader and shapes rewards
                    rewards = self.neighborhood_steps(batch, batch_p)
                # antagonistic part
                mac_loss = self.learner.train(batch=batch, t_env=t_env, episode_num=episode_num, loss_ant=loss_ant, negative_reward=True, custom_rewards=rewards)
            else:
                # antagonistic part without mimic state features
                mac_loss = self.learner.train(batch=batch, t_env=t_env, episode_num=episode_num, loss_ant=loss_ant, negative_reward=True)

        return {
            "loss_dis": loss_dis,
            "loss_ant": None if loss_ant == th.tensor(0.) else loss_ant,
            "loss_ant_mac": mac_loss,
            "accuracy_train_total": train_accuracy_tot,
            "accuracy_train_ant": train_accuracy,
            "accuracy_train_pro": train_accuracy_p,
            "accuracy_step_total": accuracy_step_tot,
            "accuracy_step_ant": accuracy_step,
            "accuracy_step_pro": accuracy_step_p
        }

    def classify_steps(self, batch: EpisodeBatch, batch_p: EpisodeBatch):
        # mimic state feature classifier
        accuracy_step_tot, accuracy_step, accuracy_step_p = None, None, None
        # select state features as inputs for classifier
        inputs = batch["state"][:, :-1, self.select_pos[-self.args.n_antagonists*2:]]
        inputs_p = batch_p["state"][:, :-1, self.select_pos[-self.args.n_antagonists*2:]]

        # get respective time step
        step_indices = th.tensor(range(0, inputs.shape[1]))
        step_indices = step_indices.repeat(inputs.shape[0]).view(inputs.shape[0], inputs.shape[1], 1)
        step_indices_p = th.tensor(range(0, inputs_p.shape[1]))
        step_indices_p = step_indices_p.repeat(inputs_p.shape[0]).view(inputs_p.shape[0], inputs_p.shape[1], 1)

        inputs = th.cat([inputs, step_indices], dim=2)
        inputs_p = th.cat([inputs_p, step_indices_p], dim=2)

        # pass through mimic state feature classifier
        predictions = self.discriminator_states(inputs)
        label_a = th.zeros(predictions.size())

        # train mimic state feature classifier according to mimic feature training scheduling
        if self.counter_train % self.args.step_classifier_interval == 0:

            predictions_p = self.discriminator_states(inputs_p)
            label_p = th.ones(predictions_p.size())

            # train classifier accuracy
            accuracy_step_tot, accuracy_step, accuracy_step_p = calc_accuracy(predictions, predictions_p)

            # BCE loss for classifier
            loss_a = self.criterion(predictions, label_a)
            loss_p = self.criterion(predictions_p, label_p)

            # backpropagation
            loss_dis = loss_p + loss_a
            self.optimiser_dis_states.zero_grad()
            loss_dis.backward()
            th.nn.utils.clip_grad_norm_(self.discriminator_states.parameters(), self.args.grad_norm_clip)
            self.optimiser_dis_states.step()

        # flip rewards (in learner do not flip again) and add reward depending on classifier output
        rewards = batch["reward"][:, :-1]
        rewards = rewards * -1

        rewards = th.where(predictions >= 0.5, rewards + 0.1, rewards - 0.1)

        return rewards, accuracy_step_tot, accuracy_step, accuracy_step_p

    def neighborhood_steps(self, batch: EpisodeBatch, batch_p: EpisodeBatch):
        # neighbourhood comparison:
        # measures distance between protagonist states and attacker states
        # if in certain epsilon range (step_neighborhood_eps) add reward else reduce reward

        # extract position from state
        state_batch = batch_p["state"][:, :-1]
        pos_p = state_batch[:, :, self.select_pos].view(batch_p.batch_size, batch_p.max_seq_length - 1, self.args.n_agents, 2)

        # preprocess positions
        antagonist_indices = list(range(self.args.n_agents - self.args.n_antagonists, self.args.n_agents))
        rewards = batch["reward"][:, :-1]
        rewards = rewards * -1
        state_batch = batch["state"][:, :-1]
        pos = state_batch[:, :, self.select_pos].view(batch.batch_size, batch.max_seq_length - 1, self.args.n_agents, 2)

        # adjust batch to have same dimensions (get rid of later states)
        min_epi = min(pos_p.shape[1], pos.shape[1])
        pos_p = pos_p[:, 0:min_epi]
        pos = pos[:, 0:min_epi]

        epi_term = th.zeros((pos.shape[0]), dtype=th.int)
        mask_reward = th.zeros(rewards.shape)

        terminated = batch["terminated"][:, :-1]
        for epi_i in range(len(terminated)):
            epi_term[epi_i] = terminated[epi_i].flatten().nonzero().item()

        # compare positioning of protagonist and attackers
        for epi_i, pos_epi in enumerate(pos):
            max_epi_mask_reward = th.zeros((pos_epi.shape[0], 1))
            for pos_p_epi in pos_p:
                mask_reward_epi = th.where(th.abs(pos_p_epi - pos_epi) <= self.move_distance, 0.1, -0.1)
                mask_reward_epi = th.sum(mask_reward_epi[:, antagonist_indices], dim=2)
                max_epi_mask_reward = th.maximum(max_epi_mask_reward, mask_reward_epi)
            max_epi_mask_reward[epi_term[epi_i]:] = 0
            max_epi_mask_reward = F.pad(max_epi_mask_reward, pad=(0, 0, 0, mask_reward.shape[1] - max_epi_mask_reward.shape[0]))
            mask_reward[epi_i] = max_epi_mask_reward

        # adjust reward
        rewards += mask_reward

        return rewards

    def make_history_batch(self, batch: EpisodeBatch):
        # makes observation history batch
        history_batch = th.zeros([batch.batch_size, batch.max_seq_length, self.args.episode_limit + 1, self.args.n_agents, self.history_dim_shape])

        for t in range(batch.max_seq_length):
            inputs = self.mac._build_inputs(batch, t=t).view(batch.batch_size, self.args.n_agents, self.mac_input_shape)

            for t1 in range(t, batch.max_seq_length):
                history_batch[:, t1, t, :, :] = inputs
        history_batch = history_batch[:, :-1]

        return history_batch

    def mac_outs(self, mac, batch: EpisodeBatch, mac_sec=None):
        # get outputs from mac and if specified from mac_sec as well (basically get outputs from protagonist and attacker)
        if mac_sec is not None:
            action_outs = []
            action_extra_outs = []
            mac.init_hidden(batch.batch_size)
            mac_sec.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                action_out = mac.forward(batch, t=t)
                action_extra_out = mac_sec.forward(batch, t=t)
                action_outs.append(action_out)
                action_extra_outs.append(action_extra_out)
            action_outs = th.stack(action_outs, dim=1)[:, :-1]
            action_extra_outs = th.stack(action_extra_outs, dim=1)[:, :-1]
            action_dis = th.stack([th.stack([t_a for t_a in t_act]) for t_act in action_outs])
            action_extra_dis = th.stack([th.stack([t_a for t_a in t_act]) for t_act in action_extra_outs])

            return action_dis, action_extra_dis
        else:
            action_outs = []
            mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                action_out = mac.forward(batch, t=t)
                action_outs.append(action_out)
            action_outs = th.stack(action_outs, dim=1)[:, :-1]
            action_dis = th.stack([th.stack([t_a for t_a in t_act]) for t_act in action_outs])

            return action_dis, None

    def test_discriminator(self, batch: EpisodeBatch):
        # set discriminator to eval mode and get respective accuracy
        self.mac.agent.eval()
        self.prey_mac.agent.eval()
        self.discriminator.eval()

        history_batch = self.make_history_batch(batch)
        history_batch_p = history_batch
        action_dis, action_dis_p = self.mac_outs(self.mac, batch, self.prey_mac)

        predictions_p = self.discriminator(action_dis_p, history_batch_p)
        predictions = self.discriminator(action_dis, history_batch)
        predictions_p = predictions_p[:, :, self.args.n_agents - self.args.n_antagonists:self.args.n_agents]
        predictions = predictions[:, :, self.args.n_agents - self.args.n_antagonists:self.args.n_agents]

        accuracy_tot, accuracy, accuracy_p = calc_accuracy(predictions, predictions_p)

        self.mac.agent.train()
        self.prey_mac.agent.train()
        self.discriminator.train()

        return accuracy_tot, accuracy, accuracy_p

    def save_models(self, path):
        # save weight parameters
        th.save(self.mac.agent.state_dict(), "{}/agent_hidden.th".format(path))
        self.mac.save_models(path)
        if self.learner.mixer is not None:
            th.save(self.learner.mixer.state_dict(), "{}/mixer_hidden.th".format(path))
        th.save(self.learner.optimiser.state_dict(), "{}/opt_hidden.th".format(path))

    def load_models(self, path):
        # load weight parameters
        self.mac.agent.load_state_dict(th.load("{}/agent_hidden.th".format(path), map_location=lambda storage, loc: storage))
        self.learner.target_mac.agent.load_state_dict(th.load("{}/agent_hidden.th".format(path), map_location=lambda storage, loc: storage))
        if self.learner.mixer is not None:
            self.learner.mixer.load_state_dict(th.load("{}/mixer_hidden.th".format(path), map_location=lambda storage, loc: storage))
        self.learner.optimiser.load_state_dict(th.load("{}/opt_hidden.th".format(path), map_location=lambda storage, loc: storage))


def calc_accuracy(predictions, predictions_prey):
    # calculate the accuracy in the predictions
    pred_bools = (predictions.detach() >= 0.5) == 0
    pred_p_bools = (predictions_prey.detach() >= 0.5) == 1

    accuracy = pred_bools.sum() / np.prod(pred_bools.size())
    accuracy_p = pred_p_bools.sum() / np.prod(pred_p_bools.size())

    accuracy_tot = (accuracy_p + accuracy) / 2

    return accuracy_tot, accuracy, accuracy_p
