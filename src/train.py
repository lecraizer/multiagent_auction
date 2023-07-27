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
        
        batch_loss = []
        for idx in range(N):
            original_actions = [agents[i].choose_action(observations[i], ep)[0] for i in range(N)]
            print('Actions: ', original_actions)

            for new_action in np.linspace(0.001, 0.999, 10):
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
            print('Bids:    ', [agents[i].choose_action(observations[0], ep)[0] for i in range(len(agents))])
            print('Rewards: ', rewards)
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

    
def MAtrainLoopCommonValue(agents, env, n_episodes, auction_type='first_price', vl=0, vh=1, eps=0.1):
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
        original_rewards = env.step(common_value, original_actions)[0]

        batch_loss = []
        for idx in range(N):
            for new_action in np.linspace(0.001, 0.999*N, 10): # test n different actions
            # or try n=100 random actions
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards = env.step(common_value, actions)
                agents[idx].remember(observations[idx], actions[idx], rewards[idx], observations[idx])
                loss = agents[idx].learn()
                batch_loss.append(loss)

        if ep % 50 == 0:
            print('\nEpisode', ep)
            print('Value:  ', common_value)
            print('Signals: ', observations)
            print('Bids:    ', original_actions)
            print('Rewards: ', original_rewards)
            for i in range(len(agents)):
                hist = manualTesting(agents[i], N, 'ag'+str(i+1), ep, n_episodes, auc_type=auction_type, vl=vl, vh=vh, eps=eps)
            literature_error.append(hist)
            loss_history.append(np.mean(batch_loss)) # bug fixed with batch_size=1
           
        # plot literature error and loss history
        plot_errors(literature_error, loss_history, N, auction_type, n_episodes)

    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])



###  Single agent training loop
def trainLoop(agent, env, n_episodes, ponderated_avg, N, BS, k):
    np.random.seed(0)
    start_time = timeit.default_timer()

    score_history = []
    for i in range(n_episodes):
        obs = env.reset()
        score = 0
        print('\nEpisode', i)
        print('Value: ', obs)
        act = agent.choose_action(obs)
        print('Bid:   ', act[0])
        new_state, reward, info = env.step(act)
        agent.remember(obs, act, reward, new_state)
        agent.learn()
        score += reward
        obs = new_state
        score_history.append(score)
        print('Score   %.2f' % score,
            'trailing ' + str(ponderated_avg) + ' games avg %.3f' % np.mean(score_history[-ponderated_avg:]))
        
        # if i % 25 == 0:
        #    agent.save_models()

    # Total training time
    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])
    return score_history
