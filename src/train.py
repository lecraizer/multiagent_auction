import timeit
import random
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

from utils import *


def decrease_learning_rate(agents, decrease_factor):
    '''
    Decrease learning rate for each neural network model
    '''
    for k in range(len(agents)):
        for param_group in agents[k].actor.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decrease_factor
        for param_group in agents[k].critic.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decrease_factor
        for param_group in agents[k].target_actor.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decrease_factor
        for param_group in agents[k].target_critic.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decrease_factor

    print('Learning rate: ', param_group['lr'])


def plot_errors(literature_error, loss_history, N, auction_type, n_episodes):
    '''
    plot literature error history and loss history
    '''
    plt.close('all')
    plt.plot(literature_error)
    plt.title('Error history')
    plt.xlabel('Episode')
    plt.ylabel('Error')
    plt.savefig('results/' + auction_type + '/N=' + str(N) + '/literature_error' + str(int(n_episodes/1000)) + 'k.png')

    plt.close('all')
    plt.plot(loss_history)
    plt.title('Loss history')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.savefig('results/' + auction_type + '/N=' + str(N) + '/loss_history' + str(int(n_episodes/1000)) + 'k.png')


def MAtrainLoop(agents, env, n_episodes, auction_type='first_price', r=1):
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
        done = False
        
        batch_loss = []
        for idx in range(N):
            original_actions = [agents[i].choose_action(observations[i], ep)[0] for i in range(N)]

            for new_action in np.linspace(0.001, 0.999, 10):
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards, done = env.step(observations, actions, r)
                agents[idx].remember(observations[idx], actions[idx], rewards[idx], observations[idx], int(done))
                loss = agents[idx].learn()
                if loss is not None:
                    batch_loss.append(loss)
                    
        done = True
        if ep % 50 == 0:
            print('\nEpisode', ep)
            print('Values:  ', observations)
            print('Bids:    ', [agents[i].choose_action(observations[0], ep)[0] for i in range(len(agents))])
            print('Rewards: ', rewards)
            for i in range(len(agents)):
                hist = manualTesting(agents[i], N, 'ag'+str(i+1), ep, n_episodes, auc_type=auction_type, r=r)
            
            literature_error.append(np.mean(hist))
            if len(batch_loss) > 0:
                loss_history.append(np.mean(batch_loss)) # bug fixed with batch_size=1
            
            decrease_factor = 0.99
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
        done = False
        
        original_actions = [agents[i].choose_action(observations[i], ep)[0] for i in range(N)]
        original_rewards = env.step(common_value, original_actions)[0]

        batch_loss = []
        for idx in range(N):
            for new_action in np.linspace(0.001, 0.999*N, 10): # test n different actions
            # or try n=100 random actions
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards, done = env.step(common_value, actions)
                agents[idx].remember(observations[idx], actions[idx], rewards[idx], observations[idx], int(done))
                loss = agents[idx].learn()
                batch_loss.append(loss)

        done = True
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
        done = False
        score = 0
        print('\nEpisode', i)
        print('Value: ', obs)
        act = agent.choose_action(obs)
        print('Bid:   ', act[0])
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
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