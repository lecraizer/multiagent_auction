import matplotlib.pyplot as plt 
import numpy as np
import math
import os


def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)


def formalize_name(auc_type):
    '''
    Formalize auction name
    '''
    auc_type = auc_type.replace('_', ' ').title()
    return auc_type


def manualTesting(agent, N, agent_name, episode, n_episodes, auc_type='first_price', r=1, max_revenue=1, eps=0.1, vl=0, vh=1):
    # reset plot variables
    plt.close('all')

    states = np.linspace(0, 1, 100)
    actions = []
    avg_error = 0
    for state in states:
        action = agent.choose_action(state, episode)[0] # bid
        if auc_type == 'first_price':
            expected_action = state*(N-1)/(N-1+r)
        elif auc_type == 'second_price':
            expected_action = state
        elif auc_type == 'tariff_discount':
            expected_action = (1-(state/max_revenue))*(N-1)/(N)
        elif auc_type == 'common_value':
            Y = ( (2*eps)/(N+1) ) * math.exp( (-N/(2*eps) )*( state-(vl+eps) ) )
            expected_action = state - eps + Y
        avg_error += abs(action - expected_action)
        actions.append(action)
    avg_error /= len(states)
    print('Average error: %.3f' % avg_error)

    # plt scatter size small
    plt.scatter(states, actions, color='black', s=0.3)
    if auc_type == 'first_price':
        plt.plot(states, states*(N-1)/(N-1+r), color='brown', linewidth=0.5)
    elif auc_type == 'second_price':
        plt.plot(states, states, color='brown', linewidth=0.5)
    elif auc_type == 'tariff_discount':
        plt.plot(states, (1-(states/max_revenue))*(N-1)/(N), color='brown', linewidth=0.5)
    elif auc_type == 'common_value':
        Y = ( (2*eps)/(N+1) ) * np.exp( (-N/(2*eps) )*( states-(vl+eps) ) )
        plt.plot(states, states - eps + Y, color='brown', linewidth=0.5)
    
    plt.title(formalize_name(auc_type) + ' Auction for ' + str(N) + ' players')
    plt.text(0.02, 0.94, 'Avg error: %.3f' % avg_error, fontsize=10, color='#696969')
    plt.legend(['Expected bid', 'Agent bid'], loc='lower right')   
    plt.xlabel('State (Value)')
    plt.ylabel('Action (Bid)')

    # set x-axis and y-axis range to [0, 1]
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])

    try:
        plt.savefig('results/' + auc_type + '/N=' + str(N) + '/' + agent_name + '_' + str(int(n_episodes/1000)) + 'k_' + 'r' + str(r) + '.png')
    except:
        os.mkdir('results/' + auc_type + '/N=' + str(N))
        plt.savefig('results/' + auc_type + '/N=' + str(N) + '/' + agent_name + '_' + str(int(n_episodes/1000)) + 'k_' + 'r' + str(r) + '.png')

    return avg_error