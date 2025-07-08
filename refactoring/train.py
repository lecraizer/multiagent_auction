import os
import random
import shutil
import timeit
import numpy as np
from utils import *
from datetime import timedelta

class TrainingMetrics:
    def __init__(self):
        self.literature_error = []
        self.loss_history = []

    def add(self, error=None, loss=None):
        if error is not None:
            self.literature_error.append(error)
        if loss is not None:
            self.loss_history.append(loss)

def get_others_states_actions(observations, original_actions, idx):
    others_observations = observations[:idx] + observations[idx+1:]
    others_actions = original_actions[:idx] + original_actions[idx+1:]
    return others_observations, others_actions

def print_episode_info(ep, value, bids, rewards, signals = None):
    print('\nEpisode', ep)
    print('Value :  ', value)
    print('Bids:    ', bids)
    print('Rewards: ', rewards)
    if signals is not None: print('Signals: ', signals)

def save_trained_agents(agents, auction_type, N, r, n_episodes):
    for i, agent in enumerate(agents):
        filename = f'{auction_type}_N_{N}_ag{i}_r{r}_{n_episodes}ep'
        agent.save_models(filename)

def copy_png_file(auction_type, N, n_episodes, ep, r):
    src = f'results/{auction_type}/N={N}/ag1_{int(n_episodes/1000)}k_r{r}.png'
    dst = 'results/.tmp/' + str(ep) + '.png'
    shutil.copy(src, dst)

def MAtrainLoop(maddpg, env, n_episodes, auction_type='first_price', r=1, max_revenue=1, gam=1, gif=False, 
                save_interval=10, tl_flag=False, extra_players=2):
    '''
    Multiagent training loop function for general auctions
    '''
    np.random.seed(0)
    start_time = timeit.default_timer()
    agents = maddpg.agents
    N = len(agents)
    grid_N = 10
    metrics = TrainingMetrics()
    for ep in range(n_episodes):
        observations = env.reset()
        original_actions = [agents[i].choose_action((observations[i]), ep)[0] for i in range(N)]
        original_rewards = env.step(observations, original_actions, r)
        batch_loss = []
        for idx in range(N):
            others_observations, others_actions = get_others_states_actions(observations, original_actions, idx)
            grid_values = [i + random.random()*(max_revenue/grid_N) for i in np.linspace(0, 0.9, grid_N)]
            for new_action in grid_values:
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards = env.step(observations, actions, r)
                maddpg.remember(observations[idx], actions[idx], rewards[idx], others_observations, others_actions)
                loss = maddpg.learn(idx, flag=tl_flag, num_tiles=extra_players)
                if loss is not None: batch_loss.append(loss)
        decrease_factor = 0.99
        if ep % save_interval == 0:
            print_episode_info(ep, observations, original_actions, original_rewards)
            hist = manualTesting(agents, N, ep, n_episodes, auc_type=auction_type, r=r, max_revenue=max_revenue, gam=gam)
            metrics.add(error = np.mean(hist))
            if len(batch_loss) > 0: metrics.add(loss = np.mean(batch_loss))
            save_trained_agents(agents, auction_type, N, r, n_episodes)
            decrease_learning_rate(agents, decrease_factor)
            plot_errors(metrics.literature_error, metrics.loss_history, N, auction_type, n_episodes)
            if gif: copy_png_file(auction_type, N, n_episodes, ep, r)
    if gif:
        create_gif()
        os.system('rm results/.tmp/*.png')
    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])

def MAtrainLoopCommonValue(maddpg, env, n_episodes, auction_type='first_price', 
                           r=1, vl=0, vh=2, eps=0.1, save_interval=10):
    '''
    Multiagent training loop function for common value auctions
    '''
    np.random.seed(0)
    start_time = timeit.default_timer()
    agents = maddpg.agents
    N = len(agents)
    metrics = TrainingMetrics()
    for ep in range(n_episodes):
        common_value, observations = env.reset()
        original_actions = [agents[i].choose_action(observations[i], ep)[0] for i in range(N)]
        original_rewards = env.step(common_value, original_actions)
        batch_loss = []
        for idx in range(N):
            others_observations, others_actions = get_others_states_actions(observations, original_actions, idx)
            for new_action in np.linspace(vl+0.001, vh-0.001, 10): # test n different actions
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards = env.step(common_value, actions)
                maddpg.remember(observations[idx], actions[idx], rewards[idx], others_observations, others_actions)
                loss = maddpg.learn(idx)
                if loss is not None: batch_loss.append(loss)
        decrease_factor = 0.999
        if ep % save_interval == 0:
            print_episode_info(ep, common_value, original_actions, original_rewards, observations)
            hist = manualTesting(agents, N, 'ag'+str(i+1), ep, n_episodes, auc_type=auction_type, vl=vl, vh=vh, eps=eps)
            metrics.add(error = hist)
            if len(batch_loss) > 0: metrics.add(loss = np.mean(batch_loss))
            save_trained_agents(agents, auction_type, N, r, n_episodes)
            decrease_learning_rate(agents, decrease_factor)
            plot_errors(metrics.literature_error, metrics.loss_history, N, auction_type, n_episodes) 

    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])

def MAtrainLoopAlternativeCommonValue(maddpg, env, n_episodes, 
                                      auction_type='first_price', 
                                      r=1, save_interval=10):
    '''
    Multiagent training loop function for common value auctions
    '''
    np.random.seed(0)
    start_time = timeit.default_timer()
    agents = maddpg.agents
    N = len(agents)
    metrics = TrainingMetrics()
    for ep in range(n_episodes):
        common_value, observations = env.reset()
        original_actions = [agents[i].choose_action(observations[i], ep)[0] for i in range(N)]
        original_rewards = env.step(common_value, original_actions)
        batch_loss = []
        for idx in range(N):
            others_observations, others_actions = get_others_states_actions(observations, original_actions, idx)
            for new_action in np.random.random(10)*N:
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards = env.step(common_value, actions)
                maddpg.remember(observations[idx], actions[idx], rewards[idx], others_observations, others_actions)
                loss = maddpg.learn(idx)
                if loss is not None: batch_loss.append(loss)
        decrease_factor = 0.999
        if ep % save_interval == 0:
            print_episode_info(ep, common_value, original_actions, original_rewards, observations)
            hist = manualTesting(agents, N, ep, n_episodes, auc_type=auction_type)
            metrics.add(error = hist)
            if len(batch_loss) > 0: metrics.add(loss = np.mean(batch_loss))
            save_trained_agents(agents, auction_type, N, r, n_episodes)
            decrease_learning_rate(agents, decrease_factor)
            plot_errors(metrics.literature_error, metrics.loss_history, N, auction_type, n_episodes)        
    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])

def MAjointtrainLoop(maddpg, env, n_episodes, auction_type='first_price', r=1, max_revenue=1, gam=1, gif=False, save_interval=10):
    '''
    Multiagent training loop function for general auctions
    '''
    np.random.seed(0)
    start_time = timeit.default_timer()
    agents = maddpg.agents
    N = len(agents)
    grid_N = 10
    metrics = TrainingMetrics()
    for ep in range(n_episodes):
        observations = env.reset()
        original_actions = [agents[i].choose_action((observations[i]), ep)[0] for i in range(N)]
        original_rewards = env.step(observations, original_actions, r)
        batch_loss = []
        for idx in range(N):
            others_observations, others_actions = get_others_states_actions(observations, original_actions, idx)
            grid_values = [i + random.random()*(1/grid_N) for i in np.linspace(0, 0.9, grid_N)]
            for new_action in grid_values:
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards = env.step(observations, actions, r)
                maddpg.remember(observations[idx], actions[idx], rewards[idx], others_observations, others_actions)
                loss = maddpg.learn(idx)
                if loss is not None: batch_loss.append(loss)
        decrease_factor = 0.99
        if ep % save_interval == 0:
            print_episode_info(ep, observations, original_actions, original_rewards)
            hist = manualTesting(agents, N, ep, n_episodes, auc_type=auction_type, 
                                     r=r, max_revenue=max_revenue, gam=gam)
            metrics.add(error = np.mean(hist))
            if len(batch_loss) > 0: metrics.add(loss = np.mean(batch_loss))
            save_trained_agents(agents, auction_type, N, r, n_episodes)
            decrease_learning_rate(agents, decrease_factor)
            plot_errors(metrics.literature_error, metrics.loss_history, N, auction_type, n_episodes)
            if gif: copy_png_file(auction_type, N, n_episodes, ep, r)
    if gif:
        create_gif()
        os.system('rm results/.tmp/*.png') 
    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])