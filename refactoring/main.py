from utils import *
from train import *
from evaluation import *
from maddpg import MADDPG
from playsound import playsound
from parameters import get_parameters

from env.first_price import *
from env.second_price import *
from env.common_value import *
from env.core_selecting import *
from env.tariff_discount import *
from env.joint_first_price import *

def get_score_history(auction, n_episodes, create_gif, N, aversion_coef, tl, extra_players, maddpg, multiagent_env):
    common_args = {
        'maddpg': maddpg,
        'env': multiagent_env,
        'n_episodes': n_episodes,
        'auction_type': auction,
        'save_interval': 50,
    }
    if auction == 'common_value':
        score_history = MAtrainLoopAlternativeCommonValue(**common_args)
    elif auction == 'joint_first_price':
        score_history = MAjointtrainLoop(**common_args, r=aversion_coef, gif=create_gif)
    elif auction == 'tariff_discount':
        score_history = MAtrainLoop(**common_args, r=aversion_coef, max_revenue=3, gif=create_gif)
    elif auction == 'core_selecting':
        score_history = MAtrainLoop(**common_args, r=aversion_coef, gam=1.0, gif=create_gif)
    else:
        score_history = MAtrainLoop(**common_args, r=aversion_coef, gif=create_gif, tl_flag=tl, extra_players=extra_players)
        
    return score_history

def copy_agent_parameters(maddpg, target_idx, source_idx):
    source = maddpg.agents[source_idx]
    target = maddpg.agents[target_idx]

    target.actor.load_state_dict(source.actor.state_dict())
    target.critic.load_state_dict(source.critic.state_dict())
    target.target_actor.load_state_dict(source.target_actor.state_dict())
    target.target_critic.load_state_dict(source.target_critic.state_dict())

def create_environment(auction, N):
    match auction:
        case 'first_price': multiagent_env = MAFirstPriceAuctionEnv(N)
        case 'second_price': multiagent_env = MASecondPriceAuctionEnv(N)
        case 'common_value': multiagent_env = MAAlternativeCommonPriceAuctionEnv(N)
        case 'tariff_discount': multiagent_env = MATariffDiscountEnv(N, max_revenue=3)
        case 'all_pay': multiagent_env = MAAllPayAuctionEnv(N)
        case 'core_selecting': multiagent_env = MACoreSelectingAuctionEnv()
        case 'joint_first_price': multiagent_env = MAJointFirstPriceAuctionEnv(N)
    return multiagent_env

def set_multiagent_env(auction, N):
    match auction:
        case 'first_price': multiagent_env = MAFirstPriceAuctionEnv(N)   
        case 'second_price': multiagent_env = MASecondPriceAuctionEnv(N)
        case 'all_pay': multiagent_env = MAAllPayAuctionEnv(N)
    return multiagent_env

def load_trained_agents(auction, maddpg, N, aversion_coef, n_episodes):
    for agent_idx in range(N):
        model_name = auction + '_N_' + str(N) + '_ag' + str(agent_idx) + '_r' + str(aversion_coef) + '_' + str(n_episodes) + 'ep'
        maddpg.agents[agent_idx].load_models(model_name)

if __name__ == "__main__":
    auction, BS, trained, n_episodes, create_gif, N, noise_std, ponderated_avg, aversion_coef, save_plot, \
    alert, tl, extra_players, z = get_parameters()

    multiagent_env = create_environment(auction, N)

    common_args_maddpg = {
        'alpha': 0.000025,
        'beta': 0.00025,
        'input_dims': 1,
        'tau': 0.001,
        'gamma': 0.99,
        'BS': BS,
        'fc1': 100, 
        'fc2': 100, 
        'n_actions': 1, 
        'total_eps': n_episodes, 
        'noise_std': 0.2,
    }
      
    maddpg = MADDPG(**common_args_maddpg, n_agents=N, tl_flag=tl, extra_players=extra_players)

    ### --- Training step --- ###
    if not trained:
        print('Training models...')
        score_history = get_score_history(auction, n_episodes, create_gif, N, aversion_coef, tl, extra_players, maddpg, 
                                          multiagent_env)   
    else: # load models
        if tl:
            new_N = N
            for i in range(extra_players-1):
                print('Transfer learning...')
                new_N = new_N + 1
                print('New number of agents:', new_N)
                ## --- Transfer learning step --- ###
                maddpg = MADDPG(**common_args_maddpg, n_agents=new_N, tl_flag=True, extra_players=extra_players-i-1) 
                load_trained_agents(auction, maddpg, N, aversion_coef, n_episodes)

                # Select random agent from the trained ones and copy its parameters to a new agent, then repeat for a new random agent
                for k in range(new_N-N):
                    copy_agent_parameters(maddpg, k+N, np.random.randint(0, N))
                
                # Now, train everyone
                print('Training models...')
                multiagent_env = set_multiagent_env(auction, new_N)
                score_history = MAtrainLoop(maddpg, multiagent_env, n_episodes, auction, r=aversion_coef, gif=create_gif, 
                                            save_interval=50, tl_flag=True, extra_players=extra_players-i-1)
                        
            # Last round of transfer learning
            print('\n\n================================\n\n')
            print('Another round of transfer learning...')
            new_N = new_N + 1
            print('New number of agents:', new_N)

            ## --- Transfer learning step --- ###
            maddpg = MADDPG(**common_args_maddpg, n_agents=new_N, tl_flag=False, extra_players=0)
            
            load_trained_agents(auction, maddpg, new_N-1, aversion_coef, n_episodes)

            # Select random agent from the trained ones and copy its parameters to a new agent
            copy_agent_parameters(maddpg, new_N-1, np.random.randint(0, new_N-1))

            # Now, train everyone
            print('Training models...')
            multiagent_env = set_multiagent_env(auction, new_N)
            score_history = MAtrainLoop(maddpg, multiagent_env, n_episodes, auction, r=aversion_coef, gif=create_gif, 
                                        save_interval=50, tl_flag=False, extra_players=0)
                
        else: # load trained agents and evaluate them
            print('Evaluating models...')
            n_samples = 100
            agents = maddpg.agents
            new_evaluate_agents(agents, n_samples, auc_type=auction)

    playsound('beep.mp3')