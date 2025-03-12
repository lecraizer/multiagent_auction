# This file contains utility functions for the auction environment such as plotting, decreasing learning rate, and manual testing.

import os
import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt 

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])

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
            param_group['lr'] *= decrease_factor
        for param_group in agents[k].critic.optimizer.param_groups:
            param_group['lr'] *= decrease_factor
        for param_group in agents[k].target_actor.optimizer.param_groups:
            param_group['lr'] *= decrease_factor
        for param_group in agents[k].target_critic.optimizer.param_groups:
            param_group['lr'] *= decrease_factor

    print('Learning rate: ', param_group['lr'])
    

def manualTesting(agents, N, episode, n_episodes, auc_type='first_price', r=1, max_revenue=1, eps=0.1, vl=0, vh=1, gam=1):
    # reset plot variables
    plt.close('all')
    
    states = np.linspace(0, 1, 100)
    if auc_type == 'tariff_discount':
        states = np.linspace(0, max_revenue, 100)

    agents_actions = []
    avg_error = 0
    for k, agent in enumerate(agents):
        actions = [] 
        for state in states:
            action = agent.choose_action(state, episode, evaluation=1)[0]  # bid
            actions.append(action)
            match auc_type:
                case 'first_price':
                    expected_action = state * (N - 1) / (N - 1 + r)
                case 'second_price':
                    expected_action = state
                case 'tariff_discount':
                    expected_action = (1 - (state / max_revenue)) * (N - 1) / N
                case 'common_value':
                    expected_action = state
                case 'all_pay':
                    expected_action = (state**N) * (N - 1) / N
                case 'core_selecting':
                    if gam == 1:
                        expected_action = state
                    else:
                        d = (np.exp(-1 + gam) - gam) / (1 - gam)
                        expected_action = 0 if state <= d else 1 + (np.log(gam + (1 - gam) * state)) / (1 - gam)
                case 'joint_first_price':
                    expected_action = state * (N - 1) / (N - 1 + r)
            avg_error += abs(action - expected_action)
        avg_error /= len(states)
        print('Avg error agent %i: %.3f' % (k, avg_error))

        agents_actions.append(actions)
        

    colors = ['#1C1B1B', '#184DB8', '#39973E', '#938D8D', '#FF7F0E', '#F15A60', '#7D3C98', '#2CA02C', '#17BECF', '#D62728']
    
    for i, agent_actions in enumerate(agents_actions):
        plt.scatter(states, agent_actions, s=8 if np.all(np.abs(agent_actions) <= 0.01) else 2,
                    label=f'Bid agent {i + 1}', color=colors[i], marker='*')
        
    count_zeros = 0
    for i in agents_actions:
        if np.all(np.abs(i) <= 0.01):
            count_zeros += 1


    # Set expected bids based on auction type
    match auc_type:
        case ['first_price', 'joint_first_price']:
            plt.plot(states, states * (N - 1) / (N - 1 + r), color='#AD1515', linewidth=1.0, label='Expected bid')
        case 'second_price':
            plt.plot(states, states, color='#AD1515', linewidth=1.0, label='Expected bid')
        case 'tariff_discount':
            plt.plot(states, (1 - (states / max_revenue)) * (N - 1) / N, color='#AD1515', linewidth=1.0, label='Expected bid')
        case 'common_value':
            plt.plot(states, states, color='#AD1515', linewidth=1.0, label='Expected bid')
        case 'all_pay':
            if N > 2:
                plt.plot(states, (states**N) * (N - 1) / N, color='#AD1515', linewidth=1.0, label='Expected bid N=%i' % N)
                # create expected bid for N=2 in a '--' line and in the same color as above but with a lower alpha
                # plt.plot(states, (states**2) * (2 - 1) / 2, color='#AD1515', linewidth=0.5, alpha=0.5, linestyle='--', label='Expected bid N=2')
                if N - count_zeros == N:
                    pass
                # elif N - count_zeros == 2:
                #     plt.plot(states, (states**2) * (2 - 1) / 2, color='#AD1515', linewidth=0.5, alpha=0.5, linestyle='--', label='Expected bid N=2')
                else:
                    remaining = N - count_zeros
                    plt.plot(states, (states**remaining) * (remaining - 1) / remaining, color='#7B14AF', linewidth=0.5, alpha=0.5, linestyle='--', label='Expected bid N=%i' % remaining)
            else:
                plt.plot(states, (states**N) * (N - 1) / N, color='#AD1515', linewidth=1.0, label='Expected bid')
        case 'core_selecting':
            if gam == 1:
                plt.plot(states, states, color='#AD1515', linewidth=1.0, label='Expected bid')
            else:
                d = (np.exp(-1 + gam) - gam) / (1 - gam)
                plt.plot(states, 0 if state <= d else 1 + (np.log(gam + (1 - gam) * states)) / (1 - gam), 
                         color='#AD1515', linewidth=1.0, label='Expected bid')

    plt.title(f'{formalize_name(auc_type)} Auction for {N} Players', fontsize=14)
    plt.xlabel('State (Value)', fontsize=14)
    plt.ylabel('Action (Bid)', fontsize=14)

    # Increase legend font size
    plt.legend(loc='upper left', fontsize=12)

    # Set x-axis and y-axis range to [0, 1]
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])

    if auc_type == 'tariff_discount': axes.set_xlim([0, max_revenue])

    # Define the directory path
    dir_path = f'results/{auc_type}/N={N}/'
    os.makedirs(dir_path, exist_ok=True)

    # Save the plot with a formatted filename
    plt.savefig(f'{dir_path}{int(n_episodes/1000)}k_r{r}.png')

    if not 'last_avg_error' in locals() or avg_error <= last_avg_error:
        plt.savefig(f'{dir_path}{int(n_episodes/1000)}k_r{r}.png')
        last_avg_error = avg_error

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