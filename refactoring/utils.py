import os
import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt

def plotLearning(auction_scores: list, filename: str, labels: list = None, window: int = 5) -> None:
    '''
    Plots the moving average of auction scores over the games and saves the generated plot.

    Args:
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

    Args:
        auc_type (str): The auction type name that needs to be formatted.

    Returns:
        auc_type (str): The formatted auction type name.
    '''
    auc_type = auc_type.replace('_', ' ').title()
    return auc_type

def decrease_learning_rate(agents: list, decrease_factor: float) -> None:
    '''
    Decreases the learning rate for each neural network model in the provided list of agents.

    Args:
        agents (list): A list of agent objects.
        decrease_factor (float): The factor which the learning rate will be multiplied to decrease it.

    Returns:
        None: The function directly modifies the learning rate of the agent models.
    '''
    for k in range(len(agents)):
        group = (agents[k].actor.optimizer.param_groups, agents[k].critic.optimizer.param_groups, 
                 agents[k].target_actor.optimizer.param_groups, agents[k].target_critic.optimizer.param_groups)
        for i in group:
            for param_group in i:
                param_group['lr'] *= decrease_factor

    print('Learning rate: ', param_group['lr'])

def calculate_expected_action(N: int, auc_type: str, state: float, r: float, max_revenue: float, gam: float) -> float:
    '''
    Calculates the expected action of agent.

    Args:
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

def calculate_agents_actions(agents: list, N: int, episode: int, auc_type: str, r: float, max_revenue: float, gam: float) -> tuple:
    '''
    Calculates the actions (bids) of each agent for a range of states and computes the average error 
    between the agent's actions and the expected bids based on auction theory.

    Args:
        agents (list): A list of agent objects.
        N (int): The number of agents.
        episode (int): The current training episode.
        auc_type (str): The type of auction.
        r (float): A parameter used in specific auction types.
        max_revenue (float): The maximum possible revenue (used in some auction types).
        gam (float): A parameter used in core-selecting auctions.

    Returns:
        tuple: 
            - states (ndarray): The array of state values used.
            - agents_actions (list): A list of action lists, one per agent.
            - avg_error (float): The average absolute error between the actions and theoretical bids.
    '''
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

    return states, agents_actions, avg_error

def plot_agents_actions(states: np.ndarray, agents_actions: list) -> None:
    '''
    Plots the actions (bids) of each agent over the given states.
    The function generates the plot but does not return any value.

    Args:
        states (ndarray): The array of states (values) used in the auction.
        agents_actions (list): A list of lists where each sublist contains the bids of an agent.
    '''
    colors = ['#1C1B1B', '#184DB8', '#39973E', '#938D8D', '#FF7F0E', '#F15A60', '#7D3C98', '#2CA02C', '#17BECF', '#D62728']
    for i, agent_actions in enumerate(agents_actions):
        small_marker = np.all(np.abs(agent_actions) <= 0.01)
        plt.scatter(states, agent_actions, s=8 if small_marker else 2,
                    label=f'Bid agent {i + 1}', color=colors[i % len(colors)], marker='*')
        
def configure_plot_layout(auc_type: str, N: int) -> plt.axes:
    '''
    Configures the layout of the auction plot, including titles, axis labels, limits, and legend.

    Args:
        auc_type (str): The type of auction.
        N (int): The number of agents.

    Returns:
        axes (plt.axes): The axes object for further customization if needed.
    '''
    plt.title(f'{formalize_name(auc_type)} Auction for {N} Players', fontsize=14)
    plt.xlabel('State (Value)', fontsize=14)
    plt.ylabel('Action (Bid)', fontsize=14)
    plt.legend(loc='upper left', fontsize=12) # Increase legend font size

    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])

    return axes

def plot_expected_bid_curve(states: np.ndarray, auc_type: str, N: int, r: float, max_revenue: float, 
                            gam: float, count_zeros: int) -> None:
    '''
    Plots the theoretical (expected) bidding curve based on auction type and parameters.
    The function adds the expected bid curve to the current plot.

    Args:
        states (ndarray): The array of states (values) used in the auction.
        auc_type (str): The type of auction.
        N (int): The number of agents.
        r (float): A parameter used in some auction types.
        max_revenue (float): The maximum possible revenue (used in some auction types).
        gam (float): A parameter used in core-selecting auctions.
        count_zeros (int): The number of agents with zero bids, used for alternate plotting logic in all-pay auctions.
    '''
    def _plot(y_vals, label_suffix='', color='#AD1515', linestyle='-', linewidth=1.0, alpha=1.0):
        plt.plot(states, y_vals, label=f'Expected bid{label_suffix}', color=color,
                 linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    match auc_type:
        case  ['first_price', 'joint_first_price']:
            _plot(states * (N - 1) / (N - 1 + r))
        case 'second_price':
            _plot(states)
        case 'tariff_discount':
            _plot((1 - (states / max_revenue)) * (N - 1) / N)
        case 'common_value':
            _plot(states)
        case 'all_pay':
            _plot((states**N) * (N - 1) / N, label_suffix=f' N={N}')
            if N > 2 and count_zeros != 0:
                remaining = N - count_zeros
                y_alt = (states**remaining) * (remaining - 1) / remaining
                _plot(y_alt, label_suffix=f' N={remaining}', color='#7B14AF', linestyle='--', linewidth=0.5, alpha=0.5)
        case 'core_selecting':
            if gam == 1:
                _plot(states)
            else:
                d = (np.exp(-1 + gam) - gam) / (1 - gam)
                y_vals = np.where(states <= d, 0, 1 + (np.log(gam + (1 - gam) * states)) / (1 - gam))
                _plot(y_vals)

def manualTesting(agents: list, N: int, episode: int, n_episodes: int, auc_type: str = 'first_price', r: float = 1, 
                  max_revenue: float = 1, eps: float = 0.1, vl: float = 0, vh: float = 1, gam: float = 1) -> float:
    '''
    Performs manual testing of agent policies by plotting their bidding behavior against the theoretical benchmark,
    and saving the resulting plot.

    Args:
        agents (list): A list of agent objects.
        N (int): The number of agents.
        episode (int): The current episode.
        n_episodes (int): Total number of episodes.
        auc_type (str): The type of auction. Default is 'first_price'.
        r (float): A parameter used in specific auction types.
        max_revenue (float): The maximum possible revenue. Default is 1.
        eps (float): Default is 0.1.
        vl (float): Default is 0.
        vh (float): Default is 1.
        gam (float): Default is 1.

    Returns:
        avg_error (float): The average error between agent bids and expected bids.
    '''
    states, agents_actions, avg_error = calculate_agents_actions(agents, N, episode, auc_type, r, max_revenue, gam)
    
    plt.close('all') # reset plot variables
    plot_agents_actions(states, agents_actions)
     
    count_zeros = 0
    for i in agents_actions:
        if np.all(np.abs(i) <= 0.01): count_zeros += 1

    # Set expected bids based on auction type
    plot_expected_bid_curve(states, auc_type, N, r, max_revenue, gam, count_zeros)
    axes = configure_plot_layout(auc_type, N)
    if auc_type == 'tariff_discount': axes.set_xlim([0, max_revenue])

    dir_path = f'results/{auc_type}/N={N}/' # Define the directory path
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(f'{dir_path}{int(n_episodes/1000)}k_r{r}.png') # Save the plot with a formatted filename

    if not 'last_avg_error' in locals() or avg_error <= last_avg_error:
        plt.savefig(f'{dir_path}{int(n_episodes/1000)}k_r{r}.png')
        last_avg_error = avg_error

    return avg_error

def save_images(type_error: str, N: int, auction_type: str, n_episodes: int) -> None:
    '''
    Saves the current plot as an image file in a specified directory, using the provided parameters to generate the file path.

    Args:
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

    Args:
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

    Args:
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