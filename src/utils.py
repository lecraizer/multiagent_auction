import matplotlib.pyplot as plt 
import numpy as np
import imageio
import glob
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
    

def manualTesting(agent, N, agent_name, episode, n_episodes, auc_type='first_price', r=1, max_revenue=1, eps=0.1, vl=0, vh=1, gam=1):
    # reset plot variables
    plt.close('all')

    # # Assymentric first price auction
    # if agent_name == 'ag1':
    #     states = np.linspace(0, 1, 100)
    # else:
    #     states = np.linspace(0, 2, 100)
    
    states = np.linspace(0, 1, 100)
    if auc_type == 'tariff_discount':
        states = np.linspace(0, max_revenue, 100)
    # elif auc_type == 'common_value':
        # states = np.linspace(vl, vh, 100)
    actions = []
    avg_error = 0
    for state in states:
        action = agent.choose_action(state, episode, evaluation=1)[0] # bid
        if auc_type == 'first_price':
            expected_action = state*(N-1)/(N-1+r)
            # # if assymetric game
            # if agent_name == 'ag1':
            #     expected_action = 4./(3*state+0.000000001) * (1-(math.sqrt(1-(3*(state**2)/4.))))
            # else:
            #     expected_action = 4./(3*state+0.000000001) * ((math.sqrt(1+(3*(state**2)/4.)))-1)
        elif auc_type == 'second_price':
            expected_action = state
        elif auc_type == 'tariff_discount':
            expected_action = (1-(state/max_revenue))*(N-1)/(N)
        elif auc_type == 'common_value':
            # Y = ( (2*eps)/(N+1) ) * math.exp( (-N/(2*eps) )*( state-(vl+eps) ) )
            # expected_action = state - eps + Y
            expected_action = state
        elif auc_type == 'all_pay': # reminder that this expected action works only for N=2
            expected_action = (state**N)*(N-1)/(N)
        elif auc_type == 'core_selecting':
            if gam == 1:
                expected_action = state
            else:
                d = (np.exp(-1+gam)-gam)/(1-gam)
                if state <= d:
                    expected_action = 0
                else:
                    expected_action = 1 + (np.log(gam+(1-gam)*state))/(1-gam)
        avg_error += abs(action - expected_action)
        actions.append(action)
    avg_error /= len(states)
    print('Average error: %.3f' % avg_error)

    # plt scatter size small
    plt.scatter(states, actions, color='black', s=0.3)
    if auc_type == 'first_price':
        plt.plot(states, states*(N-1)/(N-1+r), color='brown', linewidth=0.5)
        
        # # Assymetric first price auction
        # if agent_name == 'ag1':
        #     plt.plot(states, 4./(3*states+0.000000001) * (1-(np.sqrt(1-(3*(states**2)/4.)))), color='brown', linewidth=0.5)
        # else:
        #     plt.plot(states, 4./(3*states+0.000000001) * ((np.sqrt(1+(3*(states**2)/4.)))-1), color='brown', linewidth=0.5)
    
    elif auc_type == 'second_price':
        plt.plot(states, states, color='brown', linewidth=0.5)
    elif auc_type == 'tariff_discount':
        plt.plot(states, (1-(states/max_revenue))*(N-1)/(N), color='brown', linewidth=0.5)
    elif auc_type == 'common_value':
        plt.plot(states, states, color='brown', linewidth=0.5)
        # Y = ( (2*eps)/(N+1) ) * np.exp( (-N/(2*eps) )*( states-(vl+eps) ) )
        # plt.plot(states, states - eps + Y, color='brown', linewidth=0.5)
    elif auc_type == 'all_pay':
        plt.plot(states, (states**N)*(N-1)/(N), color='brown', linewidth=0.5)
    elif auc_type == 'core_selecting':
        if gam == 1:
            plt.plot(states, states, color='brown', linewidth=0.5)
        else:
            d = (np.exp(-1+gam)-gam)/(1-gam)
            if state <= d:
                plt.plot(states, 0, color='brown', linewidth=0.5)
            else:
                plt.plot(states, 1 + (np.log(gam+(1-gam)*states))/(1-gam), color='brown', linewidth=0.5)

        plt.plot(states, states, color='brown', linewidth=0.5)

    plt.title(formalize_name(auc_type) + ' Auction for ' + str(N) + ' players')
    plt.text(0.02, 0.94, 'Avg error: %.3f' % avg_error, fontsize=10, color='#696969')
    plt.legend(['Expected bid', 'Agent bid'], loc='lower right')   
    plt.xlabel('State (Value)')
    plt.ylabel('Action (Bid)')

    # set x-axis and y-axis range to [0, 1]
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])

    # # Assymentric first price auction
    # if agent_name == 'ag1':
    #     axes.set_xlim([0, 1])
    #     axes.set_ylim([0, 1])
    # elif agent_name == 'ag2':
    #     axes.set_xlim([0, 2])
    #     axes.set_ylim([0, 2])

    if auc_type == 'tariff_discount':
        axes.set_xlim([0, max_revenue])
    elif auc_type == 'common_value':
        # axes.set_xlim([vl, vh])
        # axes.set_ylim([vl, vh])
        # axes.set_xlim([0, 2])
        axes.set_ylim([0, 2])
    # elif auc_type == 'core_selecting':
    #     if agent_name == 'ag3':
    #         axes.set_xlim([0, 2])
    #         axes.set_ylim([0, 2])

    try:
        plt.savefig('results/' + auc_type + '/N=' + str(N) + '/' + agent_name + '_' + str(int(n_episodes/1000)) + 'k_' + 'r' + str(r) + '.png')
    except:
        os.mkdir('results/' + auc_type + '/N=' + str(N))
        plt.savefig('results/' + auc_type + '/N=' + str(N) + '/' + agent_name + '_' + str(int(n_episodes/1000)) + 'k_' + 'r' + str(r) + '.png')

    return avg_error


def plot_errors(literature_error, loss_history, N, auction_type, n_episodes):
    '''
    plot literature error history and loss history
    '''
    plt.close('all')
    plt.plot(literature_error)
    plt.title('Error history')
    plt.xlabel('Episode')
    plt.ylabel('Error')
    try:
        plt.savefig('results/' + auction_type + '/N=' + str(N) + '/literature_error' + str(int(n_episodes/1000)) + 'k.png')
    except:
        os.mkdir('results/' + auction_type + '/N=' + str(N))
        plt.savefig('results/' + auction_type + '/N=' + str(N) + '/literature_error' + str(int(n_episodes/1000)) + 'k.png')

    plt.close('all')
    plt.plot(loss_history)
    plt.title('Loss history')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    try:
        plt.savefig('results/' + auction_type + '/N=' + str(N) + '/loss_history' + str(int(n_episodes/1000)) + 'k.png')
    except:
        os.mkdir('results/' + auction_type + '/N=' + str(N))
        plt.savefig('results/' + auction_type + '/N=' + str(N) + '/loss_history' + str(int(n_episodes/1000)) + 'k.png')
  

def create_gif(img_duration=0.3):
    '''
    Create gif from png files
    '''
    input_folder = "results/.tmp/*.png"
    output_gif = "results/gifs/evolution.gif"

    # Get the list of PNG files in the input folder
    png_files = glob.glob(input_folder)

    # Read all PNG files and store them in a list
    frames = [imageio.imread(png_file) for png_file in png_files]

    print("Creating GIF from {} images".format(len(frames)))

    # Save the frames as an animated GIF
    imageio.mimsave(output_gif, frames, duration=img_duration)

    print("GIF created successfully!")