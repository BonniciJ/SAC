import random
import uuid
from argparse import ArgumentParser
from collections import deque

import gymnasium as gym
from gym.wrappers import RescaleAction
import numpy as np
import pandas as pd
import torch

from sac import SAC_Agent
from utils import MeanStdevFilter, Transition, make_gif, make_checkpoint


def train_agent_model_free(agent, env, params):
    
    update_timestep = 1
    seed = params['seed']
    log_interval = 1000
    gif_interval = 5000
    n_random_actions = params['n_random_actions']
    n_evals = params['n_evals']
    n_collect_steps = params['n_collect_steps']
    use_statefilter = params['obs_filter']
    save_model = params['save_model']
    total_steps = params['total_steps']

    save_model = True

    assert n_collect_steps > agent.batchsize, "We must initially collect as many steps as the batch size!"

    avg_length = 0
    time_step = 0
    cumulative_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    i_episode = 0
    log_episode = 0
    samples_number = 0
    episode_rewards = []
    episode_steps = []

    plot_log = []

    if use_statefilter:
        state_filter = MeanStdevFilter(env.env.observation_space.shape[0])
    else:
        state_filter = None

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_steps = env.spec.max_episode_steps


    while samples_number < total_steps:
        time_step = 0
        episode_reward = 0
        i_episode += 1
        log_episode += 1
        state = env.reset()[0]
        if state_filter:
            state_filter.update(state)
        done = False

        while (not done):
            cumulative_log_timestep += 1
            cumulative_timestep += 1
            time_step += 1
            samples_number += 1
            if samples_number < n_random_actions:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, state_filter=state_filter)
            x = env.step(action)
            nextstate, reward, done, done2, _= x
            done = done + done2
            # if we hit the time-limit, it's not a 'real' done; we don't want to assign low value to those states
            real_done = False if time_step == max_steps else done
            agent.replay_pool.push(Transition(state, action, reward, nextstate, real_done))
            state = nextstate
            if state_filter:
                state_filter.update(state)
            episode_reward += reward
            # update if it's time
            if cumulative_timestep % update_timestep == 0 and cumulative_timestep > n_collect_steps:
                q1_loss, q2_loss, pi_loss, a_loss = agent.optimize(update_timestep, state_filter=state_filter)
                if done:
                    plot_log.append([episode_reward, q1_loss, q2_loss, pi_loss, a_loss])
                n_updates += 1
            # logging
            if cumulative_timestep % log_interval == 0 and cumulative_timestep > n_collect_steps:

                avg_length = np.mean(episode_steps)
                running_reward = np.mean(episode_rewards)
                eval_reward = evaluate_agent(env, agent, state_filter, n_starts=n_evals)
                print('Episode {} \t Samples {} \t Avg length: {} \t Test reward: {} \t Train reward: {} \t Number of Policy Updates: {}'.format(i_episode, samples_number, avg_length, eval_reward, running_reward, n_updates))
                
                episode_steps = []
                episode_rewards = []

                #save the rewards to plot them later
                np.save("plot_log.npy", plot_log)

            if cumulative_timestep % gif_interval == 0:
                make_gif(agent, env, cumulative_timestep, state_filter)
                if save_model:
                    make_checkpoint(agent, cumulative_timestep)

        
        episode_steps.append(time_step)
        episode_rewards.append(episode_reward)

        

    


def evaluate_agent(env, agent, state_filter, n_starts=1):
    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state = env.reset()[0]
        while (not done):
            action = agent.get_action(state, state_filter=state_filter, deterministic=True)
            action = np.array(action) # wrap action as an array as this evironment expects this
            nextstate, reward, done, done2, _ = env.step(action)
            done = done + done2
            reward_sum += reward
            state = nextstate
    return reward_sum / n_starts


def main():
    

    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--use_obs_filter', dest='obs_filter', action='store_true')
    parser.add_argument('--update_every_n_steps', type=int, default=1)
    parser.add_argument('--n_random_actions', type=int, default=10000)
    parser.add_argument('--n_collect_steps', type=int, default=1000)
    parser.add_argument('--n_evals', type=int, default=1)
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--make_gif', dest='make_gif', action='store_true')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--total_steps', type=int, default=int(1e6))
    parser.set_defaults(obs_filter=False)
    parser.set_defaults(save_model=False)

    args = parser.parse_args()
    params = vars(args)

    seed = params['seed']


	# Change the environment here
    env = gym.make("Hopper-v5", render_mode='rgb_array')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SAC_Agent(seed, state_dim, action_dim)

    train_agent_model_free(agent=agent, env=env, params=params)


if __name__ == '__main__':
    main()
