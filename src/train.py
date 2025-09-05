import os
import random
import shutil
import timeit
import numpy as np
from utils import *
from datetime import timedelta
from gui import show_auction_episode

def get_others_states_actions(observations: list, actions: list, idx: int) -> tuple:
    """
    Extract the observations and actions of all agents except the one at the given index.

    Args:
        observations (list): List of observations for all agents.
        actions (list): List of actions taken by all agents.
        idx (int): Index of the agent to exclude.

    Returns:
        tuple: A tuple containing:
            - list: Observations of all other agents.
            - list: Actions of all other agents.
    """
    others_observations = observations[:idx] + observations[idx+1:]
    others_actions = actions[:idx] + actions[idx+1:]
    return others_observations, others_actions

def generate_grid_actions(grid_N: int, max_revenue: float) -> list:
    """
    Generate a list of grid-based bid actions with random perturbations.

    Args:
        grid_N (int): Number of grid points to generate.
        max_revenue (float): Maximum possible revenue, used to scale the perturbation.

    Returns:
        list: A list of float bid values based on a perturbed grid.
    """
    grid_values = np.linspace(0, max_revenue, grid_N)
    return [val + random.uniform(0, max_revenue / grid_N) for val in grid_values]

def log_episode(ep: int, obs: list, actions: list, rewards: list, show_gui: bool=True) -> None:
    """
    Print the values, bids, and rewards of a given episode.

    Args:
        ep (int): Episode number.
        obs (list): Observations or private values of the agents.
        actions (list): Bids submitted by the agents.
        rewards (list): Rewards received by the agents.
    """
    print(f'\nEpisode {ep}')
    print('Values:  ', obs)
    print('Bids:    ', actions)
    print('Rewards: ', rewards)

    if show_gui:
        show_auction_episode(obs, actions, rewards)

def save_models_and_update(agents: list, auction_type: str, N: int, r: float, n_episodes: int, 
                           ep: int, loss_history: list, literature_error: list, gif: bool, 
                           decrease_factor: float):
    """
    Save agent models, update learning parameters, and optionally copy image files for GIF creation.

    Args:
        agents (list): List of agents.
        auction_type (str): Type of auction being simulated.
        N (int): Number of agents.
        r (float): Reward shaping parameter.
        n_episodes (int): Total number of training episodes.
        ep (int): Current episode index.
        loss_history (list): History of loss values.
        literature_error (list): History of literature errors.
        gif (bool): Whether to create a GIF from image snapshots.
        decrease_factor (float): Factor by which to reduce learning rate.
    """
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

def MAtrainLoop(maddpg, 
                env, 
                n_episodes: int, 
                auction_type: str='first_price', 
                r: float=1, 
                t: float = 1,
                gif: bool=False, 
                save_interval: int=10,
                tl_flag: bool=False, 
                extra_players: int=2,
                show_gui: bool=False):
    """
    Multi-agent training loop for auction environments using MADDPG.

    Args:
        maddpg (MADDPG): Multi-agent DDPG trainer.
        env (AuctionEnv): Auction environment instance.
        n_episodes (int): Number of training episodes.
        auction_type (str): Type of auction.
        r (float): Reward shaping parameter.
        max_revenue (float): Maximum theoretical revenue for grid action sampling.
        gif (bool): Whether to generate GIF snapshots during training.
        save_interval (int): Interval (in episodes) at which to log and save models.
        tl_flag (bool): Whether to enable transfer learning.
        extra_players (int): Number of hypothetical agents for extended learning.
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
        original_rewards = env.step(observations, original_actions, r, t)

        batch_loss = []

        for idx in range(N):
            others_obs, others_actions = get_others_states_actions(observations, original_actions, idx)
            grid_actions = generate_grid_actions(grid_N, env.upper_bound)

            for new_action in grid_actions:
                test_actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards = env.step(observations, test_actions, r, t)
                maddpg.remember(observations[idx], test_actions[idx], rewards[idx], others_obs, others_actions)
                loss = maddpg.learn(idx, flag=(tl_flag if extra_players > 0 else False), num_tiles=extra_players)
                if loss is not None:
                    batch_loss.append(loss)
                    
        if ep % save_interval == 0:
            log_episode(ep, observations, original_actions, original_rewards, show_gui)

            hist = manualTesting(agents, N, ep, n_episodes, auc_type=auction_type, r=r, t=t,
                                 max_revenue=env.upper_bound)
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