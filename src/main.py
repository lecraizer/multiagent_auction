# Main file to run the training and evaluation of the agents in the different auction environments.

from env import *
from utils import *
from train import *
from evaluation import *
from maddpg import MADDPG
from argparser import parse_args
# from playsound import playsound

if __name__ == "__main__":

    ### --- Parsing arguments --- ###
    auction, BS, trained, n_episodes, create_gif, N, noise_std, \
    ponderated_avg, aversion_coef, save_plot, alert, \
    tl, extra_players, z = parse_args() # get parameters

    ### --- Creating environment --- ###
    if auction == 'first_price':
        multiagent_env = MAFirstPriceAuctionEnv(N)
        # multiagent_env = MAAssymetricFirstPriceAuctionEnv(N)
    elif auction == 'second_price':
        multiagent_env = MASecondPriceAuctionEnv(N)
    elif auction == 'common_value':
        multiagent_env = MAAlternativeCommonPriceAuctionEnv(N)
    elif auction == 'tariff_discount':
        max_revenue = 3
        multiagent_env = MATariffDiscountEnv(N, max_revenue=max_revenue)
    elif auction == 'all_pay':
        multiagent_env = MAAllPayAuctionEnv(N)
    elif auction == 'core_selecting':
        N = 3
        multiagent_env = MACoreSelectingAuctionEnv()
    elif auction == 'joint_first_price':
        multiagent_env = MAJointFirstPriceAuctionEnv(N)

    ### --- Creating agents --- ###    
    maddpg = MADDPG(alpha=0.000025, beta=0.00025, input_dims=1, tau=0.001,
                    gamma=0.99, BS=BS, fc1=100, fc2=100, n_actions=1, 
                    n_agents=N, total_eps=n_episodes,  noise_std=0.2, 
                    tl_flag=tl, extra_players=extra_players)


    ### --- Training step --- ###
    if not trained: # Train models if trained==False
        print('Training models...')
        if auction == 'common_value':
            score_history = MAtrainLoopAlternativeCommonValue(maddpg, multiagent_env, 
                                                              n_episodes, auction, 
                                                              save_interval=50)
        elif auction == 'joint_first_price':
            score_history = MAjointtrainLoop(maddpg, multiagent_env, n_episodes, auction, 
                                        r=aversion_coef, gif=create_gif, save_interval=50)

        elif auction == 'tariff_discount':
            score_history = MAtrainLoop(maddpg, multiagent_env, n_episodes, auction, 
                                        r=aversion_coef, max_revenue=max_revenue, 
                                        gif=create_gif, save_interval=50)
        elif auction == 'core_selecting':
            gam = 1.0
            score_history = MAtrainLoop(maddpg, multiagent_env, n_episodes, auction, 
                                        r=aversion_coef, gam=gam, gif=create_gif, save_interval=50)
        else:
            score_history = MAtrainLoop(maddpg, multiagent_env, n_episodes, auction, 
                                        r=aversion_coef, gif=create_gif, save_interval=50, tl_flag=tl, extra_players=extra_players)
        # playsound('beep.mp3') if alert else None # beep when training is done    


    # Else, load models
    else:
        print('Loading models...')
        if tl:
            new_N = N + extra_players
            print('New number of agents:', new_N)
            ## --- Transfer learning step --- ###
            print('Transfer learning...')
            maddpg = MADDPG(alpha=0.000025, beta=0.00025, input_dims=1, tau=0.001,
                            gamma=0.99, BS=BS, fc1=100, fc2=100, n_actions=1,
                            n_agents=new_N, total_eps=n_episodes, noise_std=0.2, 
                            tl_flag=False, extra_players=extra_players)
            
            # load trained agents
            for k in range(N):
                string = auction + '_N_' + str(N) + '_ag' + str(k) + '_r' + str(aversion_coef) + '_' + str(n_episodes) + 'ep'
                maddpg.agents[k].load_models(string)

            # copy first N agents' parameters to the last N agents
            for k in range(new_N-N):
                maddpg.agents[k+N].actor.load_state_dict(maddpg.agents[k].actor.state_dict())
                maddpg.agents[k+N].critic.load_state_dict(maddpg.agents[k].critic.state_dict())
                maddpg.agents[k+N].target_actor.load_state_dict(maddpg.agents[k].target_actor.state_dict())
                maddpg.agents[k+N].target_critic.load_state_dict(maddpg.agents[k].target_critic.state_dict())

            
            # train everyone in all-pay auction
            print('Training models...')
            if auction == 'first_price':
                multiagent_env = MAFirstPriceAuctionEnv(new_N)
                score_history = MAtrainLoop(maddpg, multiagent_env, n_episodes, 'first_price', 
                                            r=aversion_coef, gif=create_gif, save_interval=50, 
                                            tl_flag=False, extra_players=extra_players)
            elif auction == 'second_price':
                multiagent_env = MASecondPriceAuctionEnv(new_N)
                score_history = MAtrainLoop(maddpg, multiagent_env, n_episodes, 'second_price', 
                                            r=aversion_coef, gif=create_gif, save_interval=50, 
                                            tl_flag=False, extra_players=extra_players)
            elif auction == 'all_pay':    
                multiagent_env = MAAllPayAuctionEnv(new_N)
                score_history = MAtrainLoop(maddpg, multiagent_env, n_episodes, 'all_pay', 
                                            r=aversion_coef, gif=create_gif, save_interval=50, 
                                            tl_flag=False, extra_players=extra_players)
        
                
        else:
            ## --- Evaluation step --- ###
            print('Evaluating models...')
            n_samples = 100
            agents = maddpg.agents
            new_evaluate_agents(agents, n_samples, auc_type=auction)
