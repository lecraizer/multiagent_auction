# This file contains utility functions for the auction environment such as plotting, decreasing learning rate, and manual testing.

import matplotlib.pyplot as plt 
import numpy as np
import imageio
import glob
import os


def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.array([np.mean(scores[max(0, t-window):(t+1)]) for t in range(N)])
    x = list(range(N)) if x is None else x
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)


def formalize_name(auc_type):
    return auc_type.replace('_', ' ').title()


def decrease_learning_rate(agents, decrease_factor):
    for agent in agents:
        for opt in [agent.actor.optimizer, agent.critic.optimizer, agent.target_actor.optimizer, agent.target_critic.optimizer]:
            for group in opt.param_groups:
                group['lr'] *= decrease_factor
    print('Learning rate: ', group['lr'])  # última lida


def manualTesting(agents, N, episode, n_episodes, auc_type='first_price', r=1, max_revenue=1, eps=0.1, vl=0, vh=1, gam=1):
    plt.close('all')
    states = np.linspace(0, max_revenue if auc_type == 'tariff_discount' else 1, 100)
    colors = ['#1C1B1B', '#184DB8', '#39973E', '#938D8D', '#FF7F0E', '#F15A60', '#7D3C98', '#2CA02C', '#17BECF', '#D62728']

    avg_error = 0
    agents_actions = []

    for k, agent in enumerate(agents):
        actions = [agent.choose_action(state, episode, evaluation=1)[0] for state in states]
        agents_actions.append(actions)

        # Theoretical expected bid
        if auc_type == 'first_price':
            expected = [s * (N - 1) / (N - 1 + r) for s in states]
        elif auc_type == 'second_price':
            expected = states
        elif auc_type == 'tariff_discount':
            expected = [(1 - (s / max_revenue)) * (N - 1) / N for s in states]
        elif auc_type == 'common_value':
            expected = states
        elif auc_type == 'all_pay':
            expected = [(s**N) * (N - 1) / N for s in states]
        else:
            expected = [0 for _ in states]

        agent_error = np.mean(np.abs(np.array(actions) - np.array(expected)))
        avg_error += agent_error
        print(f'Avg error agent {k}: {agent_error:.3f}')

        marker_size = 8 if np.all(np.abs(actions) <= 0.01) else 2
        plt.scatter(states, actions, s=marker_size, label=f'Bid agent {k + 1}', color=colors[k % len(colors)], marker='*')

    avg_error /= N

    # Expected bids curve
    if auc_type == 'first_price':
        plt.plot(states, [s * (N - 1) / (N - 1 + r) for s in states], color='#AD1515', linewidth=1.0, label='Expected bid')
    elif auc_type == 'second_price':
        plt.plot(states, states, color='#AD1515', linewidth=1.0, label='Expected bid')
    elif auc_type == 'tariff_discount':
        plt.plot(states, [(1 - (s / max_revenue)) * (N - 1) / N for s in states], color='#AD1515', linewidth=1.0, label='Expected bid')
    elif auc_type == 'common_value':
        plt.plot(states, states, color='#AD1515', linewidth=1.0, label='Expected bid')
    elif auc_type == 'all_pay':
        plt.plot(states, [(s**N) * (N - 1) / N for s in states], color='#AD1515', linewidth=1.0, label=f'Expected bid N={N}')
        
        # Check for non-bidding agents and draw alternative equilibrium
        zero_bidders = sum(np.all(np.abs(a) <= 0.01) for a in agents_actions)
        active_agents = N - zero_bidders
        if 0 < active_agents < N:
            alt_exp = [(s**active_agents) * (active_agents - 1) / active_agents for s in states]
            plt.plot(states, alt_exp, color='#7B14AF', linewidth=0.5, alpha=0.5, linestyle='--', label=f'Expected bid N={active_agents}')

    plt.title(f'{formalize_name(auc_type)} Auction for {N} Players', fontsize=14)
    plt.xlabel('State (Value)', fontsize=14)
    plt.ylabel('Action (Bid)', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)

    axes = plt.gca()
    axes.set_xlim([0, max_revenue if auc_type == 'tariff_discount' else 1])
    axes.set_ylim([0, 1])

    # Save plot
    dir_path = f'results/{auc_type}/N={N}/'
    os.makedirs(dir_path, exist_ok=True)

    # Format 'r': remove decimal if whole number, otherwise replace '.' with '_'
    r_str = f"{int(r)}" if r == int(r) else f"{r}".replace('.', '_')
    fname = f"{int(n_episodes / 1000)}k_r{r_str}.png"

    plt.savefig(f"{dir_path}{fname}")
    return avg_error


def plot_errors(literature_error, loss_history, N, auction_type, n_episodes):
    plt.close('all')
    dir_path = f'results/{auction_type}/N={N}/'
    os.makedirs(dir_path, exist_ok=True)

    plt.plot(literature_error)
    plt.title('Error history')
    plt.xlabel('Episode')
    plt.ylabel('Error')
    plt.savefig(f'{dir_path}/literature_error{int(n_episodes/1000)}k.png')

    plt.close('all')
    plt.plot(loss_history)
    plt.title('Loss history')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.savefig(f'{dir_path}/loss_history{int(n_episodes/1000)}k.png')


def create_gif(img_duration=0.3):
    input_folder = "results/.tmp/*.png"
    output_gif = "results/gifs/evolution.gif"
    png_files = sorted(glob.glob(input_folder))

    frames = [imageio.imread(png) for png in png_files]
    print(f"Creating GIF from {len(frames)} images")
    imageio.mimsave(output_gif, frames, duration=img_duration)
    print("GIF created successfully!")
