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


def decrease_learning_rate(agent, decrease_factor):
    '''
    Decrease learning rate for each neural network model
    '''
    for param_group in agent.actor.optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decrease_factor
    for param_group in agent.critic.optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decrease_factor
    for param_group in agent.target_actor.optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decrease_factor
    for param_group in agent.target_critic.optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decrease_factor

    print('Learning rate: ', param_group['lr'])
    

def manualTesting(agent, N, episode, n_episodes, auc_type='first_price'):
    # reset plot variables
    plt.close('all')
    states = np.linspace(0, 1, 100)
    actions = []
    avg_error = 0
    for state in states:
        action = agent.choose_action(state, episode)[0] # bid
        if auc_type == 'first_price':
            expected_action = state*(N-1)/((N-1)+1)
        elif auc_type == 'second_price':
            expected_action = state
        elif auc_type == 'all_pay': # reminder that this expected action works only for N=2
            expected_action = (state**2.0)/2.0
        
        avg_error += abs(action - expected_action)
        actions.append(action)
    avg_error /= len(states)
    print('Average error: %.3f' % avg_error)

    # plt scatter size small
    plt.scatter(states, actions, color='black', s=0.3)
    if auc_type == 'first_price':
        plt.plot(states, states*(N-1)/((N-1)+1), color='brown', linewidth=0.5)
    elif auc_type == 'second_price':
        plt.plot(states, states, color='brown', linewidth=0.5)
    elif auc_type == 'all_pay':
        plt.plot(states, (states**2)/2, color='brown', linewidth=0.5)

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
        plt.savefig('singleagent_results/' + auc_type + '/N=' + str(N) + '/' + str(int(n_episodes/1000)) + 'k.png')
    except:
        os.mkdir('singleagent_results/' + auc_type + '/N=' + str(N))
        plt.savefig('singleagent_results/' + auc_type + '/N=' + str(N) + '/' + str(int(n_episodes/1000)) + 'k.png')

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
        plt.savefig('singleagent_results/' + auction_type + '/N=' + str(N) + '/loss_history' + str(int(n_episodes/1000)) + 'k.png')
    except:
        os.mkdir('singleagent_results/' + auction_type + '/N=' + str(N))
        plt.savefig('singleagent_results/' + auction_type + '/N=' + str(N) + '/loss_history' + str(int(n_episodes/1000)) + 'k.png')
  

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