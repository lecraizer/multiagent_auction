# Main file to run the training and evaluation of the agents in the different auction environments.

from env import *
from utils import *
from train import *
from evaluation import *
from maddpg import MADDPG
from argparser import parse_args
from playsound import playsound
import numpy as np

def create_env(auction, N, max_revenue=None):
    """Create the appropriate auction environment based on user selection."""
    if auction == 'first_price':
        return MAFirstPriceAuctionEnv(N)
    elif auction == 'second_price':
        return MASecondPriceAuctionEnv(N)
    elif auction == 'all_pay':
        return MAAllPayAuctionEnv(N)
    else:
        raise ValueError(f"Auction type '{auction}' not recognized.")

def load_agents(maddpg, auction, N, aversion_coef, n_episodes):
    """Load trained agent models."""
    for k in range(N):
        name = f"{auction}_N_{N}_ag{k}_r{aversion_coef}_{n_episodes}ep"
        maddpg.agents[k].load_models(name)

if __name__ == "__main__":

    # --- Parse arguments ---
    auction, BS, trained, n_episodes, create_gif, N, noise_std, \
    ponderated_avg, aversion_coef, save_plot, alert, \
    tl, extra_players, z = parse_args()

    # --- Create environment ---
    max_revenue = 3 if auction == 'tariff_discount' else None
    multiagent_env = create_env(auction, N, max_revenue)

    # --- Initialize agents ---
    maddpg = MADDPG(alpha=0.000025, beta=0.00025, input_dims=1, tau=0.001,
                    gamma=0.99, BS=BS, fc1=100, fc2=100, n_actions=1, 
                    n_agents=N, total_eps=n_episodes, noise_std=0.2, 
                    tl_flag=tl, extra_players=extra_players)

    # --- Train agents ---
    if not trained:
        print('Training models...')
        score_history = MAtrainLoop(maddpg, multiagent_env, n_episodes, auction, 
                                    r=aversion_coef, gif=create_gif, 
                                    save_interval=50, tl_flag=tl, extra_players=extra_players)

    # --- Transfer Learning ---
    else:
        if tl:
            # === Do not touch TL logic as per user request ===
            new_N = N
            for i in range(extra_players-1):
                print('Transfer learning...')
                new_N += 1
                print('New number of agents:', new_N)

                maddpg = MADDPG(alpha=0.000025, beta=0.00025, input_dims=1, tau=0.001,
                                gamma=0.99, BS=BS, fc1=100, fc2=100, n_actions=1,
                                n_agents=new_N, total_eps=n_episodes, noise_std=0.2, 
                                tl_flag=True, extra_players=extra_players-i-1)

                for k in range(N):
                    string = auction + '_N_' + str(N) + '_ag' + str(k) + '_r' + str(aversion_coef) + '_' + str(n_episodes) + 'ep'
                    maddpg.agents[k].load_models(string)

                for k in range(new_N - N):
                    rd_agt_idx = np.random.randint(0, N)
                    maddpg.agents[k + N].actor.load_state_dict(maddpg.agents[rd_agt_idx].actor.state_dict())
                    maddpg.agents[k + N].critic.load_state_dict(maddpg.agents[rd_agt_idx].critic.state_dict())
                    maddpg.agents[k + N].target_actor.load_state_dict(maddpg.agents[rd_agt_idx].target_actor.state_dict())
                    maddpg.agents[k + N].target_critic.load_state_dict(maddpg.agents[rd_agt_idx].target_critic.state_dict())

                multiagent_env = create_env(auction, new_N, max_revenue)

                score_history = MAtrainLoop(maddpg, multiagent_env, n_episodes, auction, 
                                            r=aversion_coef, gif=create_gif, 
                                            save_interval=50, tl_flag=True, extra_players=extra_players - i - 1)

            print('\n\n================================\n\n')
            print('Another round of transfer learning...')
            new_N += 1
            print('New number of agents:', new_N)

            maddpg = MADDPG(alpha=0.000025, beta=0.00025, input_dims=1, tau=0.001,
                            gamma=0.99, BS=BS, fc1=100, fc2=100, n_actions=1,
                            n_agents=new_N, total_eps=n_episodes, noise_std=0.2, 
                            tl_flag=False, extra_players=0)

            for k in range(new_N - 1):
                string = auction + '_N_' + str(new_N - 1) + '_ag' + str(k) + '_r' + str(aversion_coef) + '_' + str(n_episodes) + 'ep'
                maddpg.agents[k].load_models(string)

            rd_agt_idx = np.random.randint(0, new_N - 1)
            maddpg.agents[new_N - 1].actor.load_state_dict(maddpg.agents[rd_agt_idx].actor.state_dict())
            maddpg.agents[new_N - 1].critic.load_state_dict(maddpg.agents[rd_agt_idx].critic.state_dict())
            maddpg.agents[new_N - 1].target_actor.load_state_dict(maddpg.agents[rd_agt_idx].target_actor.state_dict())
            maddpg.agents[new_N - 1].target_critic.load_state_dict(maddpg.agents[rd_agt_idx].target_critic.state_dict())

            multiagent_env = create_env(auction, new_N, max_revenue)

            score_history = MAtrainLoop(maddpg, multiagent_env, n_episodes, auction,
                                        r=aversion_coef, gif=create_gif, 
                                        save_interval=50, tl_flag=True, extra_players=0)

        # --- Evaluate pre-trained agents ---
        else:
            print('Evaluating models...')
            load_agents(maddpg, auction, N, aversion_coef, n_episodes)
            evaluate_agents(maddpg.agents, n_bids=100, grid_precision=100, auc_type=auction)

    playsound('beep.mp3')
