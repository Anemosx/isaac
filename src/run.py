import datetime
import math
import sys
from functools import partial
from math import ceil
import numpy as np
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
import random
import json
import git

from dotmap import DotMap
import neptune.new as neptune

from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log, pymongo_client=None):
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}_{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), random.randint(0, 1000))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    if pymongo_client is not None:
        print("Attempting to close mongodb client")
        pymongo_client.close()
        print("Mongodb client closed")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    if not sys.platform.startswith('win'):
        os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # Init runner so we can get env info
    args.hidden_model_exists = False
    if args.train_hidden_a_learner or args.train_simultaneously or args.obvious_antagonist or args.only_gan:
        args.hidden_model_exists = True
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    # buffer for attacker
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # buffer for protagonist episodes
    buffer_p = ReplayBuffer(scheme, groups, int(args.buffer_size / args.action_imit_interval), env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # buffer for testing
    buffer_test = ReplayBuffer(scheme, groups, int((1 - args.train_test_ratio) * 500), env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    hidden_mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, hidden_mac=hidden_mac)

    args.episode_limit = runner.episode_limit
    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    # learner for attacker
    antagonist_learner = le_REGISTRY[args.learner](hidden_mac, buffer.scheme, logger, args)

    args.ally_feat_offset = runner.env.get_obs_move_feats_size() + \
                            runner.env.get_obs_enemy_feats_size()[0] * runner.env.get_obs_enemy_feats_size()[1]

    # attacker
    hidden_learner = le_REGISTRY["hidden_attacker_learner"](antagonist_learner, mac, buffer.scheme, args, runner.env)

    if args.use_cuda:
        learner.cuda()
        antagonist_learner.cuda()

    # load trained protagonist weights
    if args.load_trained and not args.train_normal_learner:
        saved_model_path = os.path.join("trained_models", args.env_args['map_name'], args.name)
        if args.load_sim_weights:
            model_path = os.path.join(saved_model_path, "simultaneous", "hidden", str(args.directory_weight))
        elif args.load_sim_obvious_weights:
            model_path = os.path.join(saved_model_path, "simultaneous", "obvious", str(args.directory_weight))
        else:
            model_path = os.path.join(saved_model_path, str(args.directory_weight))
            if args.load_t_max:
                model_path = os.path.join(model_path, "t_max")

        learner.load_models(model_path)
        args.model_load_from = model_path
        mac.agent.eval()

    # load trained attacker weights
    if args.load_hidden_trained:
        saved_hidden_model_path = os.path.join(args.local_results_path, "models", "custom", args.env_args['map_name'],
                                               args.load_hidden_path, str(args.load_hidden_epi))
        if args.load_hidden_t_max:
            saved_hidden_model_path = os.path.join(args.local_results_path, "models", "custom",
                                                   args.env_args['map_name'],
                                                   args.load_hidden_path, "t_max")
        hidden_learner.load_models(saved_hidden_model_path)
        args.model_hidden_load_from = saved_hidden_model_path
        hidden_mac.agent.eval()

    nep_logger = setup_neptune(args)

    # start training
    episode = 0
    save_t_max = True
    save_epi_max = True
    last_test_T = -args.test_interval - 1
    last_log_T = 0

    # test with dynamic attacker training ratio (only negative impact)
    if args.timed_imit:
        reward_history = None
        reward_hist_range = np.array(range(args.reward_window))
        hidden_learner.action_imit_interval = hidden_learner.action_imit_interval * 4

    # test scheduling
    test_phase_gen = phase_generator(math.ceil(args.train_test_ratio * 100), math.ceil((1 - args.train_test_ratio) * 100))
    if args.step_imit:
        protagonist_phase_gen = phase_generator(1, 1)
    else:
        protagonist_phase_gen = phase_generator(1, 0)
    pro_phase_gen_switched = False
    log_dict = {}

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # logging on neptune
    if nep_logger:
        save_path_parent = os.path.join(args.local_results_path, "models", "custom", args.env_args['map_name'],
                                 str(nep_logger._short_id))
    else:
        save_path_parent = os.path.join(args.local_results_path, "models", "custom", args.env_args['map_name'],
                                 args.unique_token)
    os.makedirs(save_path_parent, exist_ok=True)

    # save weights at mean win rates
    save_at_action_p = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    same_action_hist = []

    # adjust run episodes depending on settings
    if args.eval_runs:
        args.run_episodes = 1000
        # episode = args.run_episodes - 1000

    if args.load_hidden_trained:
        episode = args.run_episodes - 1000
        protagonist_phase_gen = phase_generator(1, 0)
        if args.sc2_replay:
            protagonist_phase_gen = phase_generator(1, 0)
            episode = args.run_episodes - 10
        if args.save_pos:
            protagonist_phase_gen = phase_generator(1, 1)
            episode = args.run_episodes - 1000
        if not args.train_normal_learner:
            args.train_hidden_a_learner = False

    while episode <= args.run_episodes or runner.t_env <= args.t_max:
        # Run for a whole episode at a time

        # finish training if defined in config
        if episode > args.run_episodes:
            if args.load_hidden_trained or args.random_actions or args.eval_runs or args.until_episodes_end:
                break

        # run eval or random attacker episodes, Random Invader
        if args.random_actions or (args.load_trained and args.load_hidden_trained) or args.eval_runs:
            # track positioning
            if args.save_pos:
                # use protagonist according to scheduling
                use_protagonist = next(protagonist_phase_gen)
                episode_batch, episode_return, win, enemy_health, ally_health, chosen_actions = runner.run(
                    test_mode=use_protagonist, use_protagonist=use_protagonist)

                position_saving(args, runner, episode_batch, use_protagonist, save_path_parent)

            else:
                use_protagonist = False
                episode_batch, episode_return, win, enemy_health, ally_health, chosen_actions = runner.run(test_mode=True)

            if use_protagonist:
                episode -= args.batch_size_run

            else:

                # log metrics
                if nep_logger:
                    nep_logger["return"].log(episode_return)
                    nep_logger["win"].log(win)
                    nep_logger["enemy_health"].log(enemy_health)
                    nep_logger["ally_health"].log(ally_health)

                # compare actions with protagonist actions
                if args.compare_actions and 0. not in chosen_actions[:, 0]:
                    chosen_actions_percentage = chosen_actions[:, 1] / chosen_actions[:, 0]
                    if nep_logger:
                        for ag_i, c_act in enumerate(chosen_actions_percentage):
                            nep_logger["ag_{}_same_a_p".format(ag_i)].log(round(c_act, 4))

                # print(f"episode {episode} return {episode_return} win {win}")

        # run original protagonist training
        elif args.train_normal_learner:
            episode_batch, episode_return, win, enemy_health, ally_health, chosen_actions = runner.run(test_mode=False)

            if nep_logger:
                nep_logger["return"].log(episode_return)
                nep_logger["win"].log(win)
                nep_logger["enemy_health"].log(enemy_health)
                nep_logger["ally_health"].log(ally_health)

            buffer.insert_episode_batch(episode_batch)
            if buffer.can_sample(args.batch_size):
                for _ in range(args.training_iters):
                    episode_sample = buffer.sample(args.batch_size)
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]
                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    mac_loss = learner.train(episode_sample, runner.t_env, episode)

                    if nep_logger:
                        nep_logger["loss_ant_mac"].log(round(mac_loss.item(), 4))

        # run attacker training
        else:
            # use protagonist according to scheduling
            use_protagonist = next(protagonist_phase_gen)
            episode_batch, episode_return, win, enemy_health, ally_health, chosen_actions = runner.run(test_mode=use_protagonist, use_protagonist=use_protagonist)

            # test episode every 10th episode
            if episode % 10 == 0 and args.log_test:
                _, episode_return_test, win_test, enemy_health_test, ally_health_test, chosen_actions_test = runner.run(test_mode=True)
                if nep_logger:
                    nep_logger["return_test"].log(episode_return_test)
                    nep_logger["win_test"].log(win_test)
                    nep_logger["enemy_health_test"].log(round(enemy_health_test, 4))
                    nep_logger["ally_health_test"].log(ally_health_test)

                # compare actions with protagonist actions
                if args.compare_actions and 0. not in chosen_actions_test[:, 0]:
                    chosen_actions_percentage_test = chosen_actions_test[:, 1] / chosen_actions_test[:, 0]
                    same_action_hist.append(sum(chosen_actions_percentage_test) / len(chosen_actions_percentage_test))
                    if len(same_action_hist) > 100:
                        same_action_hist.pop(0)

                    if nep_logger:
                        for ag_i_test, c_act_test in enumerate(chosen_actions_percentage_test):
                            nep_logger["ag_{}_same_a_p_test".format(ag_i_test)].log(round(c_act_test, 4))

            # save positioning
            if args.save_pos:
                position_saving(args, runner, episode_batch, use_protagonist, save_path_parent)

            if use_protagonist:

                episode -= args.batch_size_run
                buffer_p.insert_episode_batch(episode_batch)

            else:

                if nep_logger:
                    nep_logger["return"].log(episode_return)
                    nep_logger["win"].log(win)
                    nep_logger["enemy_health"].log(round(enemy_health, 4))
                    nep_logger["ally_health"].log(ally_health)

                # compare actions with protagonist actions
                if args.compare_actions and 0. not in chosen_actions[:, 0]:
                    chosen_actions_percentage = chosen_actions[:, 1] / chosen_actions[:, 0]
                    if nep_logger:
                        for ag_i, c_act in enumerate(chosen_actions_percentage):
                            nep_logger["ag_{}_same_a_p".format(ag_i)].log(round(c_act, 4))

                # insert batch in test or training buffer on chance
                if random.randint(0, 100) <= ((1 - args.train_test_ratio) * 100):
                    buffer_test.insert_episode_batch(episode_batch)
                else:
                    buffer.insert_episode_batch(episode_batch)

                # test episode scheduling
                test_phase = next(test_phase_gen)

                # test discriminator outputs
                if test_phase and buffer_test.can_sample(args.batch_size):

                    test_discriminator_phase(args, nep_logger, buffer_test, hidden_learner)

                # attacker training
                else:

                    if buffer.can_sample(args.batch_size) and buffer_p.can_sample(args.batch_size):
                        # if protagonist buffer is sufficient then adjust scheduling
                        if not pro_phase_gen_switched:
                            protagonist_phase_gen = phase_generator(math.ceil(args.epi_ant_pro_ratio * 100),
                                                                    math.ceil((1 - args.epi_ant_pro_ratio) * 100))
                            if args.only_gan or args.obvious_antagonist:
                                protagonist_phase_gen = phase_generator(1, 0)
                            pro_phase_gen_switched = True

                        # train attacker training_iters times
                        for _ in range(args.training_iters):

                            # sample from buffers
                            episode_sample = buffer.sample(args.batch_size)
                            max_ep_t = episode_sample.max_t_filled()
                            episode_sample = episode_sample[:, :max_ep_t]

                            episode_sample_p = buffer_p.sample(args.batch_size)
                            max_ep_t_p = episode_sample_p.max_t_filled()
                            episode_sample_p = episode_sample_p[:, :max_ep_t_p]

                            if episode_sample.device != args.device:
                                episode_sample.to(args.device)
                                episode_sample_p.to(args.device)

                            # train protagonist and attacker simultaneously (implemented but not used)
                            if args.train_simultaneously:
                                learner.train(episode_sample, runner.t_env, episode)
                                log_dict = hidden_learner.train(episode_sample, episode_sample_p, runner.t_env, episode)

                            # train ISAAC Invader, Naive Invader or Pure Imitator
                            elif args.train_hidden_a_learner:
                                log_dict = hidden_learner.train(episode_sample, episode_sample_p, runner.t_env, episode)

                            # log stuff on neptune
                            for key, value in log_dict.items():
                                if value is not None:
                                    if nep_logger:
                                        nep_logger[f"{key}"].log(round(value.item(), 4))
                                    else:
                                        print(f"{key}:", round(value.item(), 4))

        # save trained weights parameters
        if args.save_model:
            save_path = None

            # save parameters at given mean win rate
            if len(same_action_hist) >= 100 and (sum(same_action_hist[-100:]) / 100) >= save_at_action_p[0]:
                save_path = os.path.join(save_path_parent, f"same_a_{int(save_at_action_p[0] * 100)}")
                save_at_action_p.pop(0)

            # save if episode reached max
            if episode >= args.run_episodes and save_epi_max:
                save_epi_max = False
                save_path = os.path.join(save_path_parent, "epi_max")

            # save every 10k episode
            if episode % 10000 == 0 and episode > 100:
                save_path = os.path.join(save_path_parent, str(episode))

            # save if t_max time steps are reached
            if runner.t_env >= args.t_max and save_t_max:
                save_t_max = False
                save_path = os.path.join(save_path_parent, "t_max")

            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                learner.save_models(save_path)
                if args.hidden_model_exists:
                    hidden_learner.save_models(save_path)

        episode += args.batch_size_run

        # dynamic training adjust (not used as it is bad)
        if args.timed_imit:
            if reward_history is None:
                reward_history = np.array([episode_return])
            else:
                reward_history = np.append(reward_history, episode_return)
                if reward_history.shape[0] > args.reward_window:
                    reward_history = np.delete(reward_history, 0)

                    if episode % 100 == 0:
                        m, _ = np.polyfit(reward_hist_range, reward_history, 1)
                        if m >= 0:
                            hidden_learner.action_imit_interval = args.action_imit_interval
                            reward_history = None
                            args.timed_imit = False

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # save replay
    if args.sc2_replay:
        runner.save_replay()

    # save infos
    open(os.path.join(save_path_parent, "{}.txt".format(args.unique_token)), 'a').close()
    if nep_logger:
        open(os.path.join(save_path_parent, "{}.txt".format(nep_logger._short_id)), 'a').close()
    open(os.path.join(save_path_parent, "win_rate_{}.txt".format(runner.env.get_stats()['win_rate'])), 'a').close()

    args_config = vars(args)
    args_config['scheme'] = str(scheme)
    args_config['groups'] = str(groups)
    args_config['episode_limit'] = env_info["episode_limit"]
    args_config['preprocess'] = str(preprocess)
    with open(os.path.join(save_path_parent, "args.txt"), "w") as f:
        json.dump(args_config, f)

    runner.close_env()
    if nep_logger:
        nep_logger.stop()
    logger.console_logger.info("Finished Training")


# save positioning of attacker and protagonist
def position_saving(args, runner, episode_batch, use_protagonist, save_path_parent):
    select_pos = [[ag_i * runner.env.get_ally_num_attributes() + runner.env.ally_state_attr_names.index('rel_x'),
                   ag_i * runner.env.get_ally_num_attributes() + runner.env.ally_state_attr_names.index('rel_y')]
                  for ag_i in range(args.n_agents)]
    select_pos = [coord for pos in select_pos for coord in pos]

    pos_log = episode_batch["state"][:, :-1, select_pos[-args.n_antagonists * 2:]]

    step_indices = th.tensor(range(0, pos_log.shape[1]))
    step_indices = step_indices.repeat(pos_log.shape[0]).view(pos_log.shape[0], pos_log.shape[1], 1)

    pos_log = th.cat([pos_log, step_indices], dim=2)

    select_health = [
        [ag_i * runner.env.get_ally_num_attributes() + runner.env.ally_state_attr_names.index('health')]
        for ag_i in range(args.n_agents)]

    select_health = [health for sel_health in select_health for health in sel_health]

    health_epi = episode_batch["state"][:, :-1, select_health[-args.n_antagonists:]].flatten()

    if 0. in health_epi:
        pos_log = pos_log[:, :health_epi.tolist().index(0.)]

    pos_log = pos_log.view(pos_log.shape[1], pos_log.shape[2]).numpy()

    if use_protagonist:
        pos_file = os.path.join(save_path_parent, "pro_pos.npy")
    else:
        pos_file = os.path.join(save_path_parent, "ant_pos.npy")

    with open(pos_file, "ab") as pos_f:
        np.save(pos_f, pos_log)


# test discriminator outputs
def test_discriminator_phase(args, nep_logger, buffer_test, hidden_learner):
    test_sample = buffer_test.sample(args.batch_size)
    max_ep_t = test_sample.max_t_filled()
    test_sample = test_sample[:, :max_ep_t]

    if test_sample.device != args.device:
        test_sample.to(args.device)

    test_accuracy_tot, test_accuracy_a, test_accuracy_p = hidden_learner.test_discriminator(test_sample)
    if nep_logger:
        nep_logger["accuracy_test_total"].log(round(test_accuracy_tot.item(), 4))
        nep_logger["accuracy_test_ant"].log(round(test_accuracy_a.item(), 4))
        nep_logger["accuracy_test_pro"].log(round(test_accuracy_p.item(), 4))
    else:
        print("test_accuracy_tot", round(test_accuracy_tot.item(), 4))
        print("test_accuracy_ant", round(test_accuracy_a.item(), 4))
        print("test_accuracy_pro", round(test_accuracy_p.item(), 4))


# setup neptune logger
def setup_neptune(args):
    if not args.neptune_logger:
        return None
    with open('args.txt', 'w') as f:
        for key, value in vars(args).items():
            if type(value) is str:
                f.write("{}: '{}'\n".format(key, value))
            else:
                f.write("{}: {}\n".format(key, value))
    with open('neptune_auth.json', 'r') as f:
        neptune_auth = json.load(f)
        neptune_auth = DotMap(neptune_auth)
    config_nep = os.path.join(os.path.dirname(__file__), "config", "default.yaml")
    config_alg_nep = os.path.join(os.path.dirname(__file__), "config", "algs", "{}.yaml".format(args.name))
    config_env = os.path.join(os.path.dirname(__file__), "config", "envs", "{}.yaml".format(args.env))
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[0:8]
    tags = [sha, args.name, args.env_args['map_name']]
    if args.train_simultaneously:
        tags.append("sim")
    if args.obvious_antagonist:
        tags.append("obvi")
    if args.only_gan:
        tags.append("imit")
    if args.random_actions:
        tags.append("random")
    if args.train_normal_learner:
        tags.append("normal")
    if args.hidden_model_exists:
        tags.append("tr_ratio {}".format(args.action_imit_interval))
        if args.ignore_threshold != 1.0:
            tags.append("ign th {}".format(args.ignore_threshold))
        if args.step_imit:
            if args.step_neighborhood:
                tags.append("indi pos")
            if args.step_classifier:
                tags.append("step {}".format(args.step_classifier_interval))
    nep_logger = neptune.init(
        project=neptune_auth.project,
        api_token=neptune_auth.api_token,
        tags=tags,
        source_files=[config_nep, config_alg_nep, config_env, 'args.txt']
    )
    return nep_logger


# scheduling
def phase_generator(every_x: int, duration: int):
    counter = 0
    duration_counter = 0
    phase = False
    while True:
        if not phase:
            counter += 1
        if counter % every_x == 0:
            phase = True
        if phase:
            duration_counter += 1
            if duration_counter > duration:
                phase = False
                duration_counter = 0
        yield phase
