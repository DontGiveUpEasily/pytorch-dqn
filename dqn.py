import sys
import gym.spaces
import itertools
import numpy as np
import random
from collections import namedtuple
from dqn_utils import *
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
import numpy as np
import random
import torch.autograd as autograd

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

USE_CUDA = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

def learn(env,
          q_func,
          optimizer_spec,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10):
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_c, img_h, img_w)
    num_actions = env.action_space.n

    q_network = q_func(input_shape, num_actions, frame_history_len).type(Tensor)
    target_network = q_func(input_shape, num_actions, frame_history_len).type(Tensor)

    if USE_CUDA:
        q_network.cuda()
        target_network.cuda()
    optimizer = optim.RMSprop(q_network.parameters(), lr=optimizer_spec.lr_schedule.value(0))
    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000

    lr = optimizer_spec.lr_schedule.value(0)
    prev_lr = optimizer_spec.lr_schedule.value(0)
    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ret_idx = replay_buffer.store_frame(last_obs)
        encoded_obs = replay_buffer.encode_recent_observation()
        encoded_obs = encoded_obs[np.newaxis, :]
        encoded_obs = encoded_obs.astype(np.float32).transpose(0,3,1,2)
        action_probs_out = q_network(Variable(torch.from_numpy(encoded_obs)).cuda()).data.cpu().numpy()
        best_action = np.argmax(action_probs_out, axis=1)[0]
        action_probs = np.ones(num_actions) * exploration.value(t)/float(num_actions)
        action_probs[best_action] += 1 - exploration.value(t)
        this_action = np.random.choice(a=num_actions, p=action_probs)
        last_obs, reward, done, _ = env.step(this_action)
        replay_buffer.store_effect(ret_idx, this_action, reward, done)
        if done:
            last_obs = env.reset()
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            lr = optimizer_spec.lr_schedule.value(t)
            if lr != prev_lr:
                prev_lr = optimizer_spec.lr_schedule.value(t)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            # a. sample a batch of transitions
            obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask_batch = replay_buffer.sample(batch_size)
           
            obs_t_batch = obs_t_batch.transpose(0,3,1,2)
            obs_tp1_batch = obs_tp1_batch.transpose(0,3,1,2) 
            obs_t_batch = Variable(torch.from_numpy(obs_t_batch).type(Tensor))/255.0
            obs_tp1_batch = Variable(torch.from_numpy(obs_tp1_batch).type(Tensor))/255.0
            act_t_batch = Variable(torch.from_numpy(act_t_batch).type(LTensor))
            rew_t_batch = Variable(torch.from_numpy(rew_t_batch).type(Tensor))
            done_mask_batch = Variable(torch.from_numpy(done_mask_batch).type(Tensor))
            # b. train
            q_a_values = q_network(obs_t_batch).gather(1, act_t_batch.unsqueeze(1))
            q_a_values_tp1 = target_network(obs_tp1_batch).detach().max(1)[0]
            target_values = rew_t_batch + gamma * (1-done_mask_batch)*q_a_values_tp1
            loss = ( target_values - q_a_values).pow(2).sum() 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(q_network.parameters(), grad_norm_clipping)
            optimizer.step()
 
            # d. periodically update the target network 
            if num_param_updates % target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())

            num_param_updates += 1

            #####

        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            sys.stdout.flush()
