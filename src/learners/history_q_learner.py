import copy
from os import stat
from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import ConcatStateHistoryQMixer, QMixer, DoubleQMixer
import torch as th
from torch.optim import RMSprop
import numpy as np

class HistoryQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        self.state_history = getattr(args, "state_history", None)
        self.mac_input_shape = int(np.prod(mac._get_input_shape(scheme)))
        self.args.history_dim = [(self.args.episode_limit+1), args.n_agents, self.mac_input_shape]
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "qmix":
                if self.state_history == "state_history":
                    self.mixer = ConcatStateHistoryQMixer(args)
                if self.state_history == "history":
                    args.state_shape = self.args.history_dim
                    self.mixer = QMixer(args)
                if self.state_history == "hidden_state":
                    args.state_shape = [args.rnn_hidden_dim, args.n_agents]
                    self.mixer = QMixer(args)
                if self.state_history == "constrained_hidden_state":
                    self.mixer = DoubleQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, loss_ant=th.tensor(0.), negative_reward=False, custom_rewards=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if custom_rewards is not None:
            rewards = custom_rewards
        elif negative_reward:
            rewards = rewards * -1

        # Calculate estimated Q-Values
        history_batch = th.zeros([batch.batch_size, batch.max_seq_length, self.args.episode_limit+1, self.args.n_agents, self.mac_input_shape])
        hidden_states_batch = []
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t) # Shape [batch_size, n_agents, n_actions]
            mac_out.append(agent_outs)
            inputs = self.mac._build_inputs(batch, t=t).view(batch.batch_size, self.args.n_agents, self.mac_input_shape)
            hidden_states_batch.append(self.mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))
            for t1 in range(t, batch.max_seq_length):
                history_batch[:,t1,t,:,:] = inputs
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        hidden_states_batch = th.stack(hidden_states_batch, dim=1).detach()

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

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
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl
        target_hidden_states_batch = th.stack(target_hidden_states_batch, dim=1).detach()

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            if self.state_history == "state_history":
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], history_batch[:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], history_batch[:, 1:])
            elif self.state_history == "history":
                chosen_action_qvals = self.mixer(chosen_action_qvals, history_batch[:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, history_batch[:, 1:])
            elif self.state_history == "hidden_state":
                chosen_action_qvals = self.mixer(chosen_action_qvals, hidden_states_batch[:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, target_hidden_states_batch[:, 1:])
            elif self.state_history == "constrained_hidden_state":
                state_values, history_values = self.mixer(batch["actions_onehot"][:, :-1], chosen_action_qvals, batch["state"][:, :-1], hidden_states_batch[:, :-1])
                target_state_values, target_history_values = self.target_mixer(batch["actions_onehot"][:, 1:], target_max_qvals, batch["state"][:, 1:], target_hidden_states_batch[:, 1:])
                target_max_qvals = target_history_values
            else:
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        N = getattr(self.args, "n_step", 1)
        if N == 1:
            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        else:
            # N step Q-Learning targets
            n_rewards = th.zeros_like(rewards)
            gamma_tensor = th.tensor([self.args.gamma**i for i in range(N)], dtype=th.float, device=n_rewards.device)
            steps = mask.flip(1).cumsum(dim=1).flip(1).clamp_max(N).long()
            for i in range(batch.max_seq_length - 1):
                n_rewards[:,i,0] = ((rewards * mask)[:,i:i+N,0] * gamma_tensor[:(batch.max_seq_length - 1 - i)]).sum(dim=1)
            indices = th.linspace(0, batch.max_seq_length-2, steps=batch.max_seq_length-1, device=steps.device).unsqueeze(1).long()
            n_targets_terminated = th.gather(target_max_qvals*(1-terminated),dim=1,index=steps.long()+indices-1)
            targets = n_rewards + th.pow(self.args.gamma, steps.float()) * n_targets_terminated

        # Td-error
        if self.state_history == "constrained_hidden_state":
            targets_state = rewards + self.args.gamma * (1 - terminated) * target_state_values
            td_error_states = (state_values - targets_state.detach())
            td_error_histories = (history_values - targets.detach())
            mask = mask.expand_as(td_error_states)
            masked_td_error_states = td_error_states * mask
            loss_states = (masked_td_error_states ** 2).sum() / mask.sum()
            masked_td_error_history = td_error_histories * mask
            loss_history = (masked_td_error_history ** 2).sum() / mask.sum()

            constraint_error = (history_values - state_values.detach()).clamp(max=0)
            masked_constraint_error = constraint_error * mask
            loss_constraint = (masked_constraint_error ** 2).sum() / mask.sum()
            loss = loss_states + loss_history + loss_constraint + self.args.antagonist_loss * loss_ant
            masked_td_error = masked_td_error_states + masked_td_error_history
        else:
            td_error = (chosen_action_qvals - targets.detach())
            mask = mask.expand_as(td_error)
            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask
            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask.sum() + self.args.antagonist_loss * loss_ant

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            agent_utils = (th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3) * mask).sum().item() / (mask_elems * self.args.n_agents)
            self.logger.log_stat("agent_utils", agent_utils, t_env)
            self.log_stats_t = t_env
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
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
