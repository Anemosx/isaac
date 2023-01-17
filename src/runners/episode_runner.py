from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th
import random


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # save replay at better location
        if 'sc2' in self.args.env:
            self.args.env_args["replay_dir"] = self.args.replay_dir
            self.args.env_args["replay_prefix"] = f"{self.args.load_hidden_path}_{self.args.load_hidden_epi}"
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        else:
            self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, hidden_mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.hidden_mac = hidden_mac
        self.n_antagonists = self.args.n_antagonists
        self.n_protagonists = self.args.n_agents - self.args.n_antagonists
        self.dead_actions = th.tensor([[0 for _ in range(self.args.n_agents)]])

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        # reset action matching
        if self.args.compare_actions:
            self.chosen_actions = th.tensor(np.zeros((self.args.n_agents, 2)), dtype=th.int)
            self.c_action_count = th.tensor(np.zeros(self.args.n_agents), dtype=th.int)
            self.dead_time = th.tensor(np.zeros(self.args.n_agents), dtype=th.int)

    def run(self, test_mode=False, use_protagonist=False):
        self.reset()

        terminated = False
        win = 0
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        self.hidden_mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            # only select action matching if attacker is alive
            if self.args.compare_actions:
                for ally_i, ally_unit in self.env.agents.items():
                    if ally_unit.health == 0 and self.dead_time[ally_i] == 0:
                        self.dead_time[ally_i] = self.t

            # choose agent actions for episode
            if use_protagonist:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode).detach()
            else:
                actions = self.hidden_actions(test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0].cpu())
            if env_info:
                win = int(env_info['battle_won'])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state

        # choose agent actions for last step
        if use_protagonist:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode).detach()
        else:
            actions = self.hidden_actions(test_mode=test_mode)

        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        # log health
        enemy_health = 0
        for e_unit in self.env.enemies.values():
            enemy_health += e_unit.health / (e_unit.health_max * len(self.env.enemies))

        ally_health = 0
        for al_unit in self.env.agents.values():
            ally_health += al_unit.health / (al_unit.health_max * len(self.env.agents))

        # compare actions of protagonist and attacker
        if self.args.compare_actions:
            self.chosen_actions[:, 0] = self.t + 1
            for i, dead_t in enumerate(self.dead_time):
                if dead_t != 0:
                    self.chosen_actions[i, 0] = dead_t
            self.chosen_actions[:, 1] = self.c_action_count
            self.chosen_actions = self.chosen_actions[self.n_protagonists:].numpy()

            return self.batch, episode_return, win, enemy_health, ally_health, self.chosen_actions

        return self.batch, episode_return, win, enemy_health, ally_health, None

    # select actions for attackers and protagonists
    def hidden_actions(self, test_mode):
        # select protagonist actions
        if self.args.load_trained:
            actions_p = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True).detach()
        else:
            actions_p = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode).detach()

        actions = actions_p

        actions_a = None
        # select Random Invader
        if self.args.random_actions:
            avail_actions = self.batch["avail_actions"][:, self.t].tolist()[0]
            actions_a = th.tensor([[random.choice([a for a in range(len(av)) if av[a] > 0]) for av in avail_actions]])

        # select trained attacker
        elif self.args.load_hidden_trained:
            actions_a = self.hidden_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True).detach()

        # select attacker
        elif self.args.hidden_model_exists:
            actions_a = self.hidden_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode).detach()

        # substitute protagonist action with attackers
        if actions_a is not None:
            actions = th.cat((actions_p[0][:self.n_protagonists], actions_a[0][- self.n_antagonists:])).unsqueeze(0)

            # compare the substituted protagonist and attacker actions
            if self.args.compare_actions:
                self.c_action_count = th.where(actions_a == self.dead_actions, self.c_action_count - 1, self.c_action_count)
                self.c_action_count = th.where(actions_p == actions_a, self.c_action_count + 1, self.c_action_count)

        return actions

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
