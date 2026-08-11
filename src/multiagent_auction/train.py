import random
import timeit
import numpy as np
from datetime import timedelta
from multiagent_auction.utils import *

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

    # x = np.linspace(0.0, 1.0, grid_N, dtype=float)
    # w = np.sin(0.5 * np.pi * x)  # ∈ [0,1]
    # return (max_revenue * w).tolist()


def _expected_bid_curve(v, auction_type: str, n: int, t: float):
    eps = 1e-12
    if auction_type == 'first_price':
        return ((n - 1) / n) * v
    elif auction_type == 'second_price':
        return v
    elif auction_type == 'all_pay':
        return (v ** n) / n
    elif auction_type == 'partial_all_pay':
        num = (v ** n) * (n - 1) / n
        den = t + (1 - t) * (v ** (n - 1))
        return num / (den + eps)
    else:
        raise ValueError(f"Auction type '{auction_type}' not recognized.")


# --- dentro do MAtrainLoop, antes de usar:
def compute_agent_errors(agents, ep, auction_type = 'partial_all_pay', t=1, num_points=100) -> float:
    """Retorna a MÉDIA (entre agentes) do erro médio absoluto vs. lance ótimo ao longo de v em [0, upper_bound]."""
    v = np.linspace(0.0, 1.0, num_points)
    expected = _expected_bid_curve(v, auction_type, len(agents), t)

    per_agent_mae = []
    for ag in agents:
        actions = [ag.choose_action([vj], ep, evaluation=1)[0] for vj in v]
        err = float(np.mean(np.abs(np.asarray(actions, dtype=float) - expected)))
        per_agent_mae.append(err)

    return per_agent_mae



def log_episode(ep: int, obs: list, actions: list, rewards: list) -> None:
    """
    Print the values, bids, and rewards of a given episode.

    Args:
        ep (int): Episode number.
        obs (list): Observations or private values of the agents.
        actions (list): Bids submitted by the agents.
        rewards (list): Rewards received by the agents.
    """
    print(f"\n\n\nEpisode {ep}")
    print("-"*40)
    print(f"{'Player':>6} | {'Value':>8} | {'Bid':>8} | {'Reward':>8}")
    print("-"*40)

    for i, (v, a, r) in enumerate(zip(obs, actions, rewards), start=1):
        print(f"{i:6d} | {v:8.2f} | {a:8.2f} | {r:8.2f}")

    print("-"*40)
    LBL, W = 13, 10
    winner = int(np.argmax(actions)) + 1
    print(f"{'Winner:':<{LBL+4}}{'Player ' + str(winner):<{W}}")
    print(f"{'Reward:':<{LBL}}{rewards[winner-1]:>{W}.2f}")
    print(f"{'Average bid:':<{LBL}}{np.mean(actions):>{W}.2f}")
    print("-"*40)


def save_models_and_update(agents: list, auction_type: str, N: int, r: float, n_episodes: int, 
                           ep: int, loss_history: list, literature_error: list,
                           decrease_factor: float) -> None:
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
                show_gui: bool=False,
                t_list: list = None):
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
    conv_tol = 0.015 * N # tolerance for early stopping
    patience = 10 # number of consecutive episodes to meet tolerance before stopping
    last_t, fired = -1.0, set()
    ok_streak = 0  # Counter for consecutive episodes meeting the convergence tolerance

    for ep in range(n_episodes):
        # Update parameter t based on episode if t_list is provided
        if t_list is not None:
            idx = int(ep * len(t_list) / n_episodes)
            t = round(t_list[min(idx, len(t_list)-1)], 2)
            # env.t = t_list[min(idx, len(t_list)-1)]
        observations = env.reset()
        original_actions = [agents[i].choose_action(observations[i], ep)[0] for i in range(N)]
        original_rewards = env.step(observations, original_actions, r, t)

        
        # ---- EARLY STOP ---- #
        if t_list is None:  # com Transfer Learning ligado, não interrompe cedo
            errs = compute_agent_errors(agents, ep, auction_type=auction_type, t=t)
            instant_ok = all(e <= conv_tol for e in errs)
            # só começa a contar depois do mínimo de episódios
            if instant_ok:
                ok_streak += 1
                print('OK STREAK', ok_streak)
            else:
                ok_streak = 0
            # each 50 episodes, check for early stopping
            if ep % 50 == 0:
                pass
                # print(f"Episode {ep}, Errors: {[f'{e:.4f}' for e in errs]}, Streak: {ok_streak}")

            if ok_streak >= patience:
                log_episode(ep, observations, original_actions, original_rewards)
                hist = manualTesting(agents, N, ep, n_episodes, auc_type=auction_type, r=r, t=t,
                                    max_revenue=env.upper_bound)
                literature_error.append(np.mean(hist))
                save_models_and_update(agents, auction_type, N, r, n_episodes, ep,
                                    loss_history, literature_error, decrease_factor=0.99)
                print(f"[EARLY STOP] ep={ep}, streak={ok_streak}, max|erro|={max(errs):.4f} ≤ {conv_tol:.4f}")
                break
        # -------------------- #
        
                
        batch_loss = []

        for idx in range(N):
            others_obs, others_actions = get_others_states_actions(observations, original_actions, idx)
            grid_actions = generate_grid_actions(grid_N, env.upper_bound)

            for new_action in grid_actions:
                test_actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards = env.step(observations, test_actions, r, t)
                maddpg.remember(observations[idx], test_actions[idx], rewards[idx], others_obs, others_actions)
                loss = maddpg.learn(auction_type, idx, flag=(tl_flag if extra_players > 0 else False), num_tiles=extra_players)
                if loss is not None:
                    batch_loss.append(loss)


        if t_list is not None:
            for i, thr in enumerate((0.0, 1/3, 2/3)):
                if i not in fired and last_t < thr <= t:
                    manualTesting(agents, N, ep, n_episodes, auc_type=auction_type, r=r, t=t,
                                max_revenue=env.upper_bound, name=f"t{t:.2f}")
                    print(f"[CHECKPOINT] ep={ep} t={t:.2f} (thr={thr:.2f})")
                    fired.add(i)
                    break
            last_t = t  



        if ep % save_interval == 0:
            log_episode(ep, observations, original_actions, original_rewards)

            hist = manualTesting(agents, N, ep, n_episodes, auc_type=auction_type, r=r, t=t,
                                 max_revenue=env.upper_bound)
            literature_error.append(np.mean(hist))
            if batch_loss:
                loss_history.append(np.mean(batch_loss))

            save_models_and_update(agents, auction_type, N, r, n_episodes, ep,
                                   loss_history, literature_error, decrease_factor=0.99)


    elapsed_time = timeit.default_timer() - start_time
    print('\n\nTotal training time:', str(timedelta(seconds=elapsed_time)).split('.')[0])