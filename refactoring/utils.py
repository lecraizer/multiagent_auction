# This file contains utility functions for the auction environment such as plotting, decreasing learning rate, and manual testing.

import os
import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt

def plotLearning(auction_scores: list, filename: str, labels: list = None, window: int = 5) -> None:
    '''
    Plots the moving average of auction scores over the games and saves the generated plot.

    Parameters:
    auction_scores (list): A list containing the auction scores for each game.
    filename (str): The path and filename where the plot will be saved.
    labels (list): A list of labels for the x-axis. If None, it uses the game indices (0, 1, 2, ...).
    window (int): The number of games to consider for calculating the moving average. Default is 5.

    Returns:
    None: The function saves the plot as an image file and does not return any value.
    '''
    n_games = len(auction_scores)
    running_avg = np.empty(n_games)

    for t in range(n_games):
        running_avg[t] = np.mean(auction_scores[max(0, t-window):(t+1)])

    if labels is None: labels = [i for i in range(n_games)]

    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(labels, running_avg)
    plt.savefig(filename)

def formalize_name(auc_type: str) -> str:
    '''
    Formalizes the auction name by replacing underscores with spaces and capitalizing each word.

    Parameters:
    auc_type (str): The auction type name that needs to be formatted.

    Returns:
    auc_type (str): The formatted auction type name.
    '''
    auc_type = auc_type.replace('_', ' ').title()
    return auc_type

def decrease_learning_rate(agents: list, decrease_factor: float) -> None:
    '''
    Decreases the learning rate for each neural network model in the provided list of agents.

    Parameters:
    agents (list): A list of agent objects.
    decrease_factor (float): The factor which the learning rate will be multiplied to decrease it.

    Returns:
    None: The function directly modifies the learning rate of the agent models.
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

def calculate_expected_action(N: int, auc_type: str, state: float, r: float, max_revenue: float, gam: float) -> float:
    '''
    Calculates the expected action of agent.

    Parameters:
    N (int): The total number of agents.
    auc_type (str): The type of auction.
    state (float): The current state.
    r (float): A parameter used in specific auction types.
    max_revenue (float): The maximum possible revenue.
    gam (float): .

    Returns:
    float: The expected action.
    '''
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

    return expected_action
    
def manualTesting(agents, N, episode, n_episodes, auc_type='first_price', r=1, max_revenue=1, eps=0.1, vl=0, vh=1, gam=1):
    states = np.linspace(0, 1, 100)
    if auc_type == 'tariff_discount': states = np.linspace(0, max_revenue, 100)

    agents_actions = []
    avg_error = 0
    for k, agent in enumerate(agents):
        actions = [] 
        for state in states:
            action = agent.choose_action(state, episode, evaluation=1)[0]  # bid
            actions.append(action)
            expected_action = calculate_expected_action(N, auc_type, state, r, max_revenue, gam)
            avg_error += abs(action - expected_action)
        avg_error /= len(states)
        print('Avg error agent %i: %.3f' % (k, avg_error))

        agents_actions.append(actions)
        

    colors = ['#1C1B1B', '#184DB8', '#39973E', '#938D8D', '#FF7F0E', '#F15A60', '#7D3C98', '#2CA02C', '#17BECF', '#D62728']
    
    plt.close('all') # reset plot variables

    for i, agent_actions in enumerate(agents_actions):
        plt.scatter(states, agent_actions, s=8 if np.all(np.abs(agent_actions) <= 0.01) else 2,
                    label=f'Bid agent {i + 1}', color=colors[i], marker='*')
        
    count_zeros = 0
    for i in agents_actions:
        if np.all(np.abs(i) <= 0.01): count_zeros += 1

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
                if count_zeros != 0:
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

def save_images(type_error: str, N: int, auction_type: str, n_episodes: int) -> None:
    '''
    Saves the current plot as an image file in a specified directory, using the provided parameters to generate the file path.

    Parameters:
    type_error (str): The type of error (literature_error or loss_history).
    N (int): The number of agents (used in saving the plot).
    auction_type (str): The type of auction, used in the directory structure.
    n_episodes (int): The total number of episodes, used to name the file (in thousands).

    Returns:
    None: The function saves the plot as an image file.
    '''
    try:
        plt.savefig('results/' + auction_type + '/N=' + str(N) + '/' + type_error + str(int(n_episodes/1000)) + 'k.png')
    except:
        os.mkdir('results/' + auction_type + '/N=' + str(N))
        plt.savefig('results/' + auction_type + '/N=' + str(N) + '/' + type_error + str(int(n_episodes/1000)) + 'k.png')

def plot_errors(literature_error: list, loss_history: list, N: int, auction_type: str, n_episodes: int) -> None:
    '''
    Plots the literature error history and loss history over episodes and saves the resulting plots.

    Parameters:
    literature_error (list): A list containing the history of literature errors over episodes.
    loss_history (list): A list containing the history of loss values over episodes.
    N (int): The number of agents (used in saving the plot).
    auction_type (str): The type of auction, used in naming the saved plot files.
    n_episodes (int): The total number of episodes, used for naming the saved plot files (in thousands).

    Returns:
    None: The function saves the generated plots to files.
    '''
    plt.close('all')
    plt.plot(literature_error)
    plt.title('Error history')
    plt.xlabel('Episode')
    plt.ylabel('Error')
    save_images('literature_error', N, auction_type, n_episodes)

    plt.close('all')
    plt.plot(loss_history)
    plt.title('Loss history')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    save_images('loss_history', N, auction_type, n_episodes)
  

def create_gif(img_duration: float = 0.3, input_folder: str = "results/.tmp/*.png", output_gif: str = "results/gifs/evolution.gif") -> None:
    '''
    Creates an GIF from PNG image files.

    Parameters:
    img_duration (float): The duration of each frame in the GIF (in seconds). Default is 0.3 seconds.
    input_folder (str): The path to the folder containing the PNG files to be included in the GIF. Default is "results/.tmp/*.png".
    output_gif (str): The path and filename where the GIF will be saved. Default is "results/gifs/evolution.gif".

    Returns:
    None: The function creates and saves the GIF.
    '''
    png_files = glob.glob(input_folder) # Get the list of PNG files in the input folder
    frames = [imageio.imread(png_file) for png_file in png_files] # Read all PNG files and store them in a list

    print("Creating GIF from {} images".format(len(frames)))

    imageio.mimsave(output_gif, frames, duration=img_duration) # Save the frames as an animated GIF

    print("GIF created successfully!")