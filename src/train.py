import timeit
import random
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import shutil
import os
from utils import *


def MAtrainLoop(agents, env, n_episodes, auction_type='first_price', r=1, max_revenue=1, gif=False, save_interval=10):
    '''
    Multiagent training loop function for general auctions
    '''
    np.random.seed(0)
    start_time = timeit.default_timer()
    N = len(agents)
    literature_error = []
    loss_history = []
    for ep in range(n_episodes):
        observations = env.reset()

        original_actions = [agents[i].choose_action(observations[i], ep)[0] for i in range(N)]
        original_rewards = env.step(observations, original_actions, r)
        
        batch_loss = []
        for idx in range(N):          
            for new_action in np.linspace(0.001, max_revenue-0.001, 500):
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards = env.step(observations, actions, r)
                agents[idx].remember(observations[idx], actions[idx], rewards[idx])
                loss = agents[idx].learn()
                if loss is not None:
                    batch_loss.append(loss)

        decrease_factor = 0.999
        if ep % save_interval == 0:
            print('\nEpisode', ep)
            print('Values:  ', observations)
            print('Bids:    ', original_actions)
            print('Rewards: ', original_rewards)
            for i in range(len(agents)):
                hist = manualTesting(agents[i], N, 'ag'+str(i+1), ep, n_episodes, auc_type=auction_type, r=r, max_revenue=max_revenue)
            
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

    
def MAtrainLoopCommonValue(agents, env, n_episodes, auction_type='first_price', r=1, vl=0, vh=1, eps=0.1, save_interval=10):
    '''
    Multiagent training loop function for common value auctions
    '''
    np.random.seed(0)
    start_time = timeit.default_timer()
    N = len(agents)

    literature_error = []
    loss_history = []
    for ep in range(n_episodes):
        common_value, observations = env.reset()
        
        original_actions = [agents[i].choose_action(observations[i], ep)[0] for i in range(N)]
        original_rewards = env.step(common_value, original_actions)

        batch_loss = []
        for idx in range(N):
            for new_action in np.linspace(vl+0.001, vh-0.001, 10): # test n different actions
            # or try n=100 random actions
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards = env.step(common_value, actions)
                agents[idx].remember(observations[idx], actions[idx], rewards[idx])
                loss = agents[idx].learn()
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



###  Single agent training loop
def trainLoop(agent, env, n_episodes, N=2, auction_type='first_price', save_interval=100):
    np.random.seed(0)
    start_time = timeit.default_timer()

    score_history = []
    for ep in range(n_episodes):
        obs = env.reset()
        act = agent.choose_action(obs, ep)
        reward = env.step(act)
        agent.remember(obs, act, reward)
        agent.learn()
        # score += reward
        # obs = new_state
        # score_history.append(score)
        # print('Score   %.2f' % score,
            # 'trailing ' + str(ponderated_avg) + ' games avg %.3f' % np.mean(score_history[-ponderated_avg:]))
        
        if ep % save_interval == 0:
            print('\nEpisode:', ep)
            print('Value:  ', obs)
            print('Bid:    ', act[0])
            print('Reward: ', reward)
            manualTesting(agent, N, ep, n_episodes, auc_type=auction_type)

    # Total training time
    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])
