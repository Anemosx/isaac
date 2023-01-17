# From https://github.com/wjh720/QPLEX/, added here for convenience.
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer, ConcatStateHistoryDMAQer, DoubleDMAQer
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop
import numpy as np


class HistoryDMAQ_qattenLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        self.state_history = getattr(args, "state_history", None)
        self.mac_input_shape = int(np.prod(mac._get_input_shape(scheme)))
        self.args.history_dim = [(self.args.episode_limit+1), args.n_agents, self.mac_input_shape]
        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args)
                if self.state_history == "state_history":
                    self.mixer = ConcatStateHistoryDMAQer(args)
                if self.state_history == "history":
                    args.state_shape = self.args.history_dim
                    self.mixer = DMAQer(args)
                if self.state_history == "hidden_state":
                    args.state_shape = [args.rnn_hidden_dim, args.n_agents]
                    self.mixer = DMAQer(args)
                if self.state_history == "constrained_hidden_state":
                    self.mixer = DoubleDMAQer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_actions = self.args.n_actions

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,
                  show_demo=False, save_data=None, loss_ant=th.tensor(0.), negative_reward=False, custom_rewards=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        if custom_rewards is not None:
            rewards = custom_rewards
        elif negative_reward:
            rewards = rewards * -1

        # Calculate estimated Q-Values
        history_batch = th.zeros([batch.batch_size, batch.max_seq_length, self.args.episode_limit+1, self.args.n_agents, self.mac_input_shape])
        hidden_states_batch = []
        mac_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            inputs = self.mac._build_inputs(batch, t=t).view(batch.batch_size, self.args.n_agents, self.mac_input_shape)
            hidden_states_batch.append(self.mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))
            for t1 in range(t, batch.max_seq_length):
                history_batch[:,t1,t,:,:] = inputs
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        hidden_states_batch = th.stack(hidden_states_batch, dim=1).detach()

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()
            # self.logger.log_stat('agent_1_%d_q_1' % save_data[0], np.squeeze(q_data)[0], t_env)
            # self.logger.log_stat('agent_2_%d_q_2' % save_data[1], np.squeeze(q_data)[1], t_env)

        # Calculate the Q-Values necessary for the target
        target_hidden_states_batch = []
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_hidden_states_batch.append(self.target_mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        target_hidden_states_batch = th.stack(target_hidden_states_batch, dim=1).detach()

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,))#.cuda()
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if mixer is not None:
            if self.state_history == "state_history":
                ans_chosen = mixer(chosen_action_qvals, batch["state"][:, :-1], history_batch[:, :-1], is_v=True)
                ans_adv = mixer(chosen_action_qvals, batch["state"][:, :-1], history_batch[:, :-1], actions=actions_onehot,
                                    max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            elif self.state_history == "hidden_state":
                ans_chosen = mixer(chosen_action_qvals, hidden_states_batch[:, :-1], is_v=True)
                ans_adv = mixer(chosen_action_qvals, hidden_states_batch[:, :-1], actions=actions_onehot,
                                    max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            elif self.state_history == "constrained_hidden_state":
                state_values, history_values = mixer(chosen_action_qvals, batch["state"][:, :-1], hidden_states_batch[:, :-1], is_v=True)
                target_state_values, target_history_values = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target_hidden_states_batch[:, 1:])
                target_max_qvals = target_history_values
            else:
                ans_chosen = mixer(chosen_action_qvals, history_batch[:, :-1], is_v=True)
                ans_adv = mixer(chosen_action_qvals, history_batch[:, :-1], actions=actions_onehot,
                                    max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                if self.state_history == "state_history":
                    target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], history_batch[:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], history_batch[:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                elif self.state_history == "hidden_state":
                    target_chosen = self.target_mixer(target_chosen_qvals, target_hidden_states_batch[:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals, target_hidden_states_batch[:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mixer(target_chosen_qvals, history_batch[:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals, history_batch[:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                if self.state_history == "state_history":
                    target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], history_batch[:, 1:], is_v=True)
                elif self.state_history == "hidden_state":
                    target_max_qvals = self.target_mixer(target_max_qvals, target_hidden_states_batch[:, 1:], is_v=True)
                else:
                    target_max_qvals = self.target_mixer(target_max_qvals, history_batch[:, 1:], is_v=True)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum() + self.args.antagonist_loss * loss_ant

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env
        return loss

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None, loss_ant=th.tensor(0.), negative_reward=False, custom_rewards=None):
        loss = self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                       show_demo=show_demo, save_data=save_data, loss_ant=loss_ant, negative_reward=negative_reward, custom_rewards=custom_rewards)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        return loss

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
