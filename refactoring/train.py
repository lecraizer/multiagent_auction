import os
import random
import shutil
import timeit
import numpy as np
from utils import *
from datetime import timedelta

def get_others_states_actions(observations, original_actions, idx):
    others_observations = observations[:idx] + observations[idx+1:]
    others_actions = original_actions[:idx] + original_actions[idx+1:]
    return others_observations, others_actions

def generate_grid_actions(grid_N, max_revenue):
    grid_values = np.linspace(0, 0.9, grid_N)
    return [val + random.uniform(0, max_revenue / grid_N) for val in grid_values]

def log_episode(ep, obs, actions, rewards):
    print(f'\nEpisode {ep}')
    print('Values:  ', obs)
    print('Bids:    ', actions)
    print('Rewards: ', rewards)

def save_models_and_update(agents, auction_type, N, r, n_episodes, ep, loss_history, literature_error, gif, decrease_factor):
    for k, agent in enumerate(agents):
        model_name = f"{auction_type}_N_{N}_ag{k}_r{r}_{n_episodes}ep"
        agent.save_models(model_name)
    
    decrease_learning_rate(agents, decrease_factor)
    plot_errors(literature_error, loss_history, N, auction_type, n_episodes)

    if gif:
        src = f'results/{auction_type}/N={N}/ag1_{int(n_episodes / 1000)}k_r{r}.png'
        dst = f'results/.tmp/{ep}.png'
        if os.path.exists(src):
            shutil.copy(src, dst)

def MAtrainLoop(maddpg, env, n_episodes, auction_type='first_price',
                r=1, max_revenue=1, gam=1, gif=False, save_interval=10,
                tl_flag=False, extra_players=2):
    """
    Multiagent training loop function for general auctions
    """
    np.random.seed(0)
    start_time = timeit.default_timer()
    
    agents = maddpg.agents
    N = len(agents)
    grid_N = 10
    loss_history, literature_error = [], []

    for ep in range(n_episodes):
        observations = env.reset()
        original_actions = [agents[i].choose_action(observations[i], ep)[0] for i in range(N)]
        original_rewards = env.step(observations, original_actions, r)

        batch_loss = []

        for idx in range(N):
            others_obs, others_actions = get_others_states_actions(observations, original_actions, idx)
            grid_actions = generate_grid_actions(grid_N, max_revenue)

            for new_action in grid_actions:
                test_actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards = env.step(observations, test_actions, r)
                maddpg.remember(observations[idx], test_actions[idx], rewards[idx], others_obs, others_actions)
                loss = maddpg.learn(idx, flag=(tl_flag if extra_players > 0 else False), num_tiles=extra_players)
                if loss is not None:
                    batch_loss.append(loss)
                    
        if ep % save_interval == 0:
            log_episode(ep, observations, original_actions, original_rewards)

            hist = manualTesting(agents, N, ep, n_episodes, auc_type=auction_type, r=r,
                                 max_revenue=max_revenue, gam=gam)
            literature_error.append(np.mean(hist))
            if batch_loss:
                loss_history.append(np.mean(batch_loss))

            save_models_and_update(agents, auction_type, N, r, n_episodes, ep,
                                   loss_history, literature_error, gif, decrease_factor=0.99)

    if gif:
        create_gif()
        os.system('rm results/.tmp/*.png')

    elapsed_time = timeit.default_timer() - start_time
    print('\n\nTotal training time:', str(timedelta(seconds=elapsed_time)).split('.')[0])