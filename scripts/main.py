import os
import sys
import time
import copy
import math
import collections
from tqdm import tqdm

import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
sys.path.append('./')
sys.path.append('..')
from scripts.create_agent import createAgent
from utils.parameters import *
from storage.buffer import QLearningBufferExpert, QLearningBuffer
from helping_hands_rl_envs import env_factory
from utils.logger import Logger
from utils.schedules import LinearSchedule
from utils.torch_utils import rand_perlin_2d
from utils.env_wrapper import EnvWrapper

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def addPerlinNoiseToObs(obs, c):
    for i in range(num_processes):
        obs[i, 0] += (c * rand_perlin_2d((90, 90), (
            int(np.random.choice([1, 2, 3, 5, 6], 1)[0]),
            int(np.random.choice([1, 2, 3, 5, 6], 1)[0]))) + c)

def addPerlinNoiseToInHand(in_hand, c):
    if in_hand_mode != 'proj':
        for i in range(num_processes):
            in_hand[i, 0] += (c * rand_perlin_2d((24, 24), (
                int(np.random.choice([1, 2], 1)[0]),
                int(np.random.choice([1, 2], 1)[0]))) + c)

def getCurrentObs(in_hand, obs):
    obss = []
    for i, o in enumerate(obs):
        obss.append((o.squeeze(), in_hand[i].squeeze()))
    return obss


def train_step(agent, replay_buffer, logger):
    batch = replay_buffer.sample(batch_size)
    loss, td_error = agent.update(batch)
    logger.trainingBookkeeping(loss, td_error.mean().item())
    logger.num_training_steps += 1
    if logger.num_training_steps % target_update_freq == 0:
        agent.updateTarget()

def saveModelAndInfo(logger, agent):
    logger.saveModel(logger.num_steps, env, agent)
    logger.saveLearningCurve(1000)
    logger.saveLossCurve(100)
    logger.saveTdErrorCurve(100)
    logger.saveRewards()
    logger.saveLosses()
    logger.saveTdErrors()
    logger.saveStepLeftCurve(1000)

def train():
    start_time = time.time()
    if seed is not None:
        set_seed(seed)
    # setup env
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)

    # setup agent
    agent = createAgent()

    if load_model_pre:
        agent.loadModel(load_model_pre)
    agent.train()

    # logging
    simulator_str = copy.copy(simulator)
    if simulator == 'pybullet':
        simulator_str += ('_' + robot)
    log_dir = os.path.join(log_pre, '{}'.format(alg))
    if note:
        log_dir += '_'
        log_dir += note

    logger = Logger(log_dir, env, 'train', num_processes, max_episode, log_sub)
    hyper_parameters['model_shape'] = agent.getModelStr()
    logger.saveParameters(hyper_parameters)

    if buffer_type == 'expert':
        replay_buffer = QLearningBufferExpert(buffer_size)
    else:
        replay_buffer = QLearningBuffer(buffer_size)
    exploration = LinearSchedule(schedule_timesteps=explore, initial_p=init_eps, final_p=final_eps)

    states, in_hands, obs = envs.reset()

    if load_sub:
        logger.loadCheckPoint(os.path.join(log_dir, load_sub, 'checkpoint'), envs, agent, replay_buffer)

    # pre train
    if load_buffer is not None and not load_sub:
        logger.loadBuffer(replay_buffer, load_buffer, load_n)
    if pre_train_step > 0:
        pbar = tqdm(total=pre_train_step)
        while len(logger.losses) < pre_train_step:
            t0 = time.time()
            train_step(agent, replay_buffer, logger)
            if logger.num_training_steps % 1000 == 0:
                logger.saveLossCurve(100)
                logger.saveTdErrorCurve(100)
            if not no_bar:
                pbar.set_description('loss: {:.3f}, time: {:.2f}'.format(float(logger.getCurrentLoss()), time.time()-t0))
                pbar.update(len(logger.losses)-pbar.n)

            if (time.time() - start_time) / 3600 > time_limit:
                logger.saveCheckPoint(args, envs, agent, replay_buffer)
                exit(0)
        pbar.close()
        logger.saveModel(0, 'pretrain', agent)
        # agent.sl = sl

    if not no_bar:
        pbar = tqdm(total=max_episode)
        pbar.set_description('Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()

    obs = obs.permute(0, 3, 1, 2)
    in_hands = in_hands.permute(0, 3, 1, 2)

    while logger.num_episodes < max_episode:
        # add noise
        if perlin:
          addPerlinNoiseToObs(obs, perlin)
          addPerlinNoiseToInHand(in_hands, perlin)

        if fixed_eps:
            if logger.num_episodes < planner_episode:
                eps = 1
            else:
                eps = final_eps
        else:
            eps = exploration.value(logger.num_episodes)
        if planner_episode > logger.num_episodes:
            if np.random.random() < eps:
                is_expert = 1
                plan_actions = envs.getNextAction()
                actions_star_idx, actions_star = agent.getActionFromPlan(plan_actions)
            else:
                is_expert = 0
                q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs, final_eps)
        else:
            is_expert = 0
            q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs, eps)

        if alg.find('dagger')>=0:
            plan_actions = envs.getNextAction()
            planner_actions_star_idx, planner_actions_star = agent.getActionFromPlan(plan_actions)

        buffer_obs = getCurrentObs(in_hands, obs)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        envs.stepAsync(actions_star, auto_reset=False)

        if len(replay_buffer) >= training_offset:
            for training_iter in range(training_iters):
                train_step(agent, replay_buffer, logger)

        states_, in_hands_, obs_, rewards, dones = envs.stepWait()
        steps_lefts = envs.getStepLeft()
        obs_ = obs_.permute(0, 3, 1, 2)
        in_hands_ = in_hands_.permute(0, 3, 1, 2)

        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_states_, reset_in_hands_, reset_obs_ = envs.reset_envs(done_idxes)
            reset_obs_ = reset_obs_.permute(0, 3, 1, 2)
            reset_in_hands_ = reset_in_hands_.permute(0, 3, 1, 2)
            for j, idx in enumerate(done_idxes):
                states_[idx] = reset_states_[j]
                in_hands_[idx] = reset_in_hands_[j]
                obs_[idx] = reset_obs_[j]

        buffer_obs_ = getCurrentObs(in_hands_, obs_)

        for i in range(num_processes):
            if alg.find('dagger') >= 0:
                replay_buffer.add(
                    ExpertTransition(states[i], buffer_obs[i], planner_actions_star_idx[i], rewards[i], states_[i],
                                     buffer_obs_[i], dones[i], steps_lefts[i], torch.tensor(is_expert)))
            else:
                replay_buffer.add(
                    ExpertTransition(states[i], buffer_obs[i], actions_star_idx[i], rewards[i], states_[i],
                                     buffer_obs_[i], dones[i], steps_lefts[i], torch.tensor(is_expert))
                )
        logger.stepBookkeeping(rewards.numpy(), steps_lefts.numpy(), dones.numpy())

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = copy.copy(in_hands_)

        if (time.time() - start_time)/3600 > time_limit:
            break

        if not no_bar:
            timer_final = time.time()
            description = 'Steps:{}; Reward:{:.03f}; Explore:{:.02f}; Loss:{:.03f}; Time:{:.03f}'.format(
                logger.num_steps, logger.getCurrentAvgReward(1000), eps, float(logger.getCurrentLoss()),
                timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_episodes-pbar.n)
        logger.num_steps += num_processes

        if logger.num_steps % (num_processes * save_freq) == 0:
            saveModelAndInfo(logger, agent)

    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(args, envs, agent, replay_buffer)
    envs.close()

if __name__ == '__main__':
    train()