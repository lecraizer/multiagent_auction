import timeit
import random
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import shutil
import os
from utils import *


def MAtrainLoop(maddpg, env, n_episodes, auction_type='first_price', r=1, max_revenue=1, gam=1, gif=False, save_interval=10):
    '''
    Multiagent training loop function for general auctions
    '''
    np.random.seed(0)
    start_time = timeit.default_timer()
    agents = maddpg.agents
    N = len(agents)
    literature_error = []
    loss_history = []
    for ep in range(n_episodes):
        observations = env.reset()
        original_actions = [agents[i].choose_action(observations[i], ep)[0] for i in range(N)]
        original_rewards = env.step(observations, original_actions, r)
        batch_loss = []
        for idx in range(N):
            others_observations = observations[:idx] + observations[idx+1:]
            others_actions = original_actions[:idx] + original_actions[idx+1:]
            # for new_action in np.linspace(0.001, max_revenue-0.001, 10):
            for new_action in np.random.random(10)*max_revenue:
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards = env.step(observations, actions, r)
                
                maddpg.remember(observations[idx], actions[idx], rewards[idx], others_observations, others_actions)
                loss = maddpg.learn()
                if loss is not None:
                    batch_loss.append(loss)

        decrease_factor = 0.99
        if ep % save_interval == 0:
            print('\nEpisode', ep)
            print('Values:  ', observations)
            print('Bids:    ', original_actions)
            print('Rewards: ', original_rewards)
            for i in range(len(agents)):
                hist = manualTesting(agents[i], N, 'ag'+str(i+1), ep, n_episodes, auc_type=auction_type, 
                                     r=r, max_revenue=max_revenue, gam=gam)
            
            literature_error.append(np.mean(hist))
            if len(batch_loss) > 0:
                loss_history.append(np.mean(batch_loss))
            
            # save models each n episodes
            for k, agent in enumerate(agents):
                string = auction_type + '_ag' + str(k) + '_r' + str(r) + '_' + str(n_episodes) + 'ep'
                agents[k].save_models(string)

            # decrease learning rate each n episodes
            decrease_learning_rate(agents, decrease_factor)

            # # plot literature error and loss history
            plot_errors(literature_error, loss_history, N, auction_type, n_episodes)

            if gif:
                png_file = 'results/' + auction_type + '/N=' + str(N) + '/' + 'ag1_' + str(int(n_episodes/1000)) + 'k_' + 'r' + str(r) + '.png'
                destination_file = 'results/.tmp/' + str(ep) + '.png'
                shutil.copy(png_file, destination_file)

    if gif:
        create_gif()
        os.system('rm results/.tmp/*.png') # remove images from results/images_for_gif

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

    literature_error = []
    loss_history = []
    for ep in range(n_episodes):
        common_value, observations = env.reset()
        original_actions = [agents[i].choose_action(observations[i], ep)[0] for i in range(N)]
        original_rewards = env.step(common_value, original_actions)
        batch_loss = []
        for idx in range(N):
            others_observations = observations[:idx] + observations[idx+1:]
            others_actions = original_actions[:idx] + original_actions[idx+1:]
            for new_action in np.linspace(vl+0.001, vh-0.001, 10): # test n different actions
            # or try n=100 random actions
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards = env.step(common_value, actions)
                maddpg.remember(observations[idx], actions[idx], rewards[idx], observations[(idx+1)%2], actions[(idx+1)%2])
                loss = maddpg.learn()
                if loss is not None:
                    batch_loss.append(loss)
        
        
        decrease_factor = 0.999
        if ep % save_interval == 0:
            print('\nEpisode', ep)
            print('Value:  ', common_value)
            print('Signals: ', observations)
            print('Bids:    ', original_actions)
            print('Rewards: ', original_rewards)
            for i in range(len(agents)):
                hist = manualTesting(agents[i], N, 'ag'+str(i+1), ep, n_episodes, auc_type=auction_type, vl=vl, vh=vh, eps=eps)
            
            literature_error.append(hist)
            if len(batch_loss) > 0:
                loss_history.append(np.mean(batch_loss)) # bug fixed with batch_size=1

            # save models each n episodes
            for k, agent in enumerate(agents):
                string = auction_type + '_ag' + str(k) + '_r' + str(r) + '_' + str(n_episodes) + 'ep'
                agents[k].save_models(string)

            # decrease learning rate each n episodes
            decrease_learning_rate(agents, decrease_factor)

            # plot literature error and loss history
            plot_errors(literature_error, loss_history, N, auction_type, n_episodes)        

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

    literature_error = []
    loss_history = []
    for ep in range(n_episodes):
        common_value, observations = env.reset()
        
        original_actions = [agents[i].choose_action(observations[i], ep)[0] for i in range(N)]
        
        # original_actions[1] = observations[1] # player 2 in Nash Equilibrium trial

        original_rewards = env.step(common_value, original_actions)

        batch_loss = []
        for idx in range(N):
            # if idx == 1:
            #     continue
            for new_action in np.linspace(0, N, 10): # test n different actions
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards = env.step(common_value, actions)
                maddpg.remember(observations[idx], actions[idx], rewards[idx], observations[(idx+1)%2], actions[(idx+1)%2])
                loss = maddpg.learn()
                if loss is not None:
                    batch_loss.append(loss)
        
        decrease_factor = 0.999
        if ep % save_interval == 0:
            print('\nEpisode', ep)
            print('Value:  ', common_value)
            print('Signals: ', observations)
            print('Bids:    ', original_actions)
            print('Rewards: ', original_rewards)
            for i in range(len(agents)):
                hist = manualTesting(agents[i], N, 'ag'+str(i+1), ep, n_episodes, 
                                     auc_type=auction_type)
            
            literature_error.append(hist)
            if len(batch_loss) > 0:
                loss_history.append(np.mean(batch_loss)) # bug fixed with batch_size=1

            # save models each n episodes
            for k, agent in enumerate(agents):
                string = auction_type + '_ag' + str(k) + '_r' + str(r) + '_' + str(n_episodes) + 'ep'
                agents[k].save_models(string)

            # decrease learning rate each n episodes
            decrease_learning_rate(agents, decrease_factor)

            # plot literature error and loss history
            plot_errors(literature_error, loss_history, N, auction_type, n_episodes)        

    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])