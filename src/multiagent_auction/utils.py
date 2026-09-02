import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def calculate_all_pay_optimal_bid(states, n_agents, r):
    '''
    Calculates the optimal bid for the all-pay auction using the ivp solver.
    '''
    states = np.asarray(states, dtype=float)

    if np.isclose(r, 1.0):
        return ((n_agents - 1) / n_agents) * states**n_agents

    eps = 1e-6

    def ode(v, y):
        b = max(float(y[0]), eps)
        surplus = max(v - b, eps)

        numerator = (
            ((n_agents - 1) / r)
            * v**(n_agents - 2)
            * (surplus**r + b**r)
        )

        denominator = (
            v**(n_agents - 1) * surplus**(r - 1)
            + (1 - v**(n_agents - 1)) * b**(r - 1)
        )

        return [numerator / denominator]

    positive_states = states[states >= eps]

    solution = solve_ivp(
        ode,
        (eps, float(states.max())),
        [((n_agents - 1) / n_agents) * eps**n_agents],
        t_eval=positive_states
    )

    expected = np.zeros_like(states)
    expected[states >= eps] = solution.y[0]

    return expected


def calculate_asymmetric_first_price_bid(states, r):
    '''
    Calculates the asymmetric first-price equilibrium for agents with
    heterogeneous risk-aversion coefficients.

    Assumptions:
        - Independent private values
        - Uniform values on [0, 1]
        - Utility u_i(x) = x^r_i

    Args:
        states (np.ndarray): Values at which the bidding strategies are evaluated.
        r (list): Risk aversion coefficient for each agent.

    Returns:
        list: One optimal bidding curve for each agent.
    '''
    states = np.asarray(states, dtype=float)
    r = np.asarray(r, dtype=float)

    N = len(r)
    eps = 1e-5
    singular_eps = 1e-6

    def integrate_backward(max_bid):

        def ode(b, phi):
            denominator = phi - b

            A = r / denominator
            common_term = np.sum(A) / (N - 1)

            return phi * (common_term - A)

        def singularity_event(b, phi):
            return np.min(phi - b) - singular_eps

        singularity_event.terminal = True
        singularity_event.direction = 0

        solution = solve_ivp(
            ode,
            (max_bid, eps),
            np.ones(N),
            events=singularity_event,
            rtol=1e-7,
            atol=1e-9,
            max_step=0.005
        )

        reached_zero = (
            solution.success
            and solution.t[-1] <= eps * 1.001
        )

        return reached_zero, solution

    # Find the equilibrium maximum bid using bisection.
    low = eps * 10
    high = 1 - eps

    valid, best_solution = integrate_backward(low)

    if not valid:
        raise RuntimeError(
            "Could not initialize asymmetric first-price solver."
        )

    for _ in range(35):

        middle = (low + high) / 2

        valid, solution = integrate_backward(middle)

        if valid:
            low = middle
            best_solution = solution
        else:
            high = middle

    max_bid = low

    valid, solution = integrate_backward(max_bid)

    if not valid:
        solution = best_solution

    # The solver integrates from max_bid -> 0.
    # Reverse arrays so that bids increase from 0 -> max_bid.
    bid_grid = solution.t[::-1]
    phi = solution.y[:, ::-1]

    expected = []

    # Invert phi_i(b) numerically to obtain b_i(v).
    for i in range(N):

        values = np.concatenate(
            ([0.0], phi[i], [1.0])
        )

        bids = np.concatenate(
            ([0.0], bid_grid, [max_bid])
        )

        order = np.argsort(values)

        values = values[order]
        bids = bids[order]

        values, unique_indices = np.unique(
            values,
            return_index=True
        )

        bids = bids[unique_indices]

        curve = np.interp(
            states,
            values,
            bids
        )

        expected.append(curve)

    return expected


def plotLearning(auction_scores: list[float], filename: str, labels: list = None,
                 window: int = 5) -> None:
    '''
    Plots the moving average of auction scores over the games and saves the generated plot.

    Args:
        auction_scores (list[float]): A list containing the auction scores for each game.
        filename (str): The path and filename where the plot will be saved.
        labels (list): A list of labels for the x-axis. If None, it uses the game indices.
        window (int): The number of games to consider for calculating the moving average.
                      Default is 5.
    '''
    n_games = len(auction_scores)
    running_avg = np.array([
        np.mean(auction_scores[max(0, t-window):(t+1)])
        for t in range(n_games)
    ])

    labels = list(range(n_games)) if labels is None else labels

    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(labels, running_avg)
    plt.savefig(filename)


def formalize_name(auc_type: str) -> str:
    '''
    Formalizes the auction name by replacing underscores with spaces and
    capitalizing each word.

    Args:
        auc_type (str): The auction type name that needs to be formatted.

    Returns:
        str: The formatted auction type name.
    '''
    return auc_type.replace('_', ' ').title()


def decrease_learning_rate(agents: list, decrease_factor: float) -> None:
    '''
    Decreases the learning rate for each neural network model in the provided
    list of agents.

    Args:
        agents (list): A list of agent objects.
        decrease_factor (float): The factor which the learning rate will be
                                 multiplied to decrease it.
    '''
    for agent in agents:
        for opt in [
            agent.actor.optimizer,
            agent.critic.optimizer,
            agent.target_actor.optimizer,
            agent.target_critic.optimizer
        ]:
            for group in opt.param_groups:
                group['lr'] *= decrease_factor

    print(f"Learning Rate: {group['lr']:.6f}")


def calculate_expected_action(n_agents: int, auc_type: str, states: np.ndarray, r: float, t: float,
                              max_revenue: float) -> list:
    '''
    Calculates the expected action of agent.

    Args:
        N (int): The total number of agents.
        auc_type (str): The type of auction.
        states (np.ndarray): The current state.
        r (float): A parameter used in specific auction types.
        max_revenue (float): The maximum possible revenue.

    Returns:
        list: The expected action.
    '''
    if auc_type == 'first_price':
        expected = [
            s * (n_agents - 1) / (n_agents - 1 + r)
            for s in states
        ]

    elif auc_type == 'second_price':
        expected = states

    elif auc_type == 'tariff_discount':
        expected = [
            (1 - (s / max_revenue)) * (n_agents - 1) / n_agents
            for s in states
        ]

    elif auc_type == 'common_value':
        expected = states

    elif auc_type == 'all_pay':
        expected = calculate_all_pay_optimal_bid(
            states,
            n_agents,
            r
        )

    elif auc_type == 'partial_all_pay':
        numerator = [
            (v**n_agents) * (n_agents - 1) / n_agents
            for v in states
        ]

        denominator = [
            t + (1 - t) * (v**(n_agents - 1))
            for v in states
        ]

        expected = [
            num / den
            for num, den in zip(numerator, denominator)
        ]

    else:
        expected = [0 for _ in states]

    return expected


def calculate_asymmetric_expected_action(n_agents: int, auc_type: str, states: np.ndarray, r: list,
                                         t: float, max_revenue: float) -> list:
    '''
    Calculates one expected bidding curve for each risk aversion coefficient.

    Args:
        n_agents (int): The total number of agents.
        auc_type (str): The type of auction.
        states (np.ndarray): The current state.
        r (list): Risk aversion coefficient for each agent.
        t (float): A parameter used in specific auction types.
        max_revenue (float): The maximum possible revenue.

    Returns:
        list: A list containing one expected bidding curve for each agent.
    '''

    if auc_type == 'first_price':
        return calculate_asymmetric_first_price_bid(
            states,
            r
        )

    expected = []

    for r_i in r:
        expected.append(
            calculate_expected_action(
                n_agents,
                auc_type,
                states,
                r_i,
                t,
                max_revenue
            )
        )

    return expected


def calculate_agents_actions(agents: list, N: int, episode: int, auc_type: str, r: list, t: float,
                             max_revenue: float) -> tuple:
    '''
    Calculates the actions (bids) of each agent for a range of states and computes the average error
    between the agent's actions and the expected bids based on auction theory.

    Args:
        agents (list): A list of agent objects.
        N (int): The number of agents.
        episode (int): The current training episode.
        auc_type (str): The type of auction.
        r (list): A parameter used in specific auction types.
        max_revenue (float): The maximum possible revenue (used in some auction types).

    Returns:
        tuple:
            - states (ndarray): The array of state values used.
            - agents_actions (list): A list of action lists, one per agent.
            - avg_error (float): The average absolute error between the actions and theoretical bids.
    '''
    states = np.linspace(0, max_revenue, 100)
    avg_error = 0
    agents_actions = []

    asymmetric = not np.allclose(r, r[0])

    if asymmetric:
        expected_actions = calculate_asymmetric_expected_action(
            N,
            auc_type,
            states,
            r,
            t,
            max_revenue
        )

    for k, agent in enumerate(agents):
        actions = [
            agent.choose_action(
                state,
                episode,
                evaluation=1
            )[0]
            for state in states
        ]

        agents_actions.append(actions)

        if asymmetric:
            expected_action = expected_actions[k]

        else:
            expected_action = calculate_expected_action(
                N,
                auc_type,
                states,
                r[0],
                t,
                max_revenue
            )

        agent_error = np.mean(
            np.abs(
                np.array(actions)
                - np.array(expected_action)
            )
        )

        avg_error += agent_error

        print(
            f'Avg error agent {k + 1}: '
            f'{agent_error:.2f}'
        )

    print("-"*40)

    return states, agents_actions, avg_error/N


def plot_agents_actions(states: np.ndarray, agents_actions: list) -> None:
    '''
    Plots the actions (bids) of each agent over the given states.
    The function generates the plot but does not return any value.

    Args:
        states (ndarray): The array of states (values) used in the auction.
        agents_actions (list): A list of lists where each sublist contains the bids of an agent.
    '''
    colors = [
        '#1C1B1B',
        '#184DB8',
        '#39973E',
        '#938D8D',
        '#FF7F0E',
        '#F15A60',
        '#7D3C98',
        '#2CA02C',
        '#17BECF',
        '#D62728'
    ]

    for i, actions in enumerate(agents_actions):
        marker_size = (
            8
            if np.all(np.abs(actions) <= 0.01)
            else 2
        )

        plt.scatter(
            states,
            actions,
            s=marker_size,
            label=f'Bid agent {i + 1}',
            color=colors[i % len(colors)],
            marker='*'
        )


def configure_plot_layout(auc_type: str, N: int, max_revenue: float):
    '''
    Configures the layout of the auction plot, including titles, axis labels,
    limits, and legend.

    Args:
        auc_type (str): The type of auction.
        N (int): The number of agents.
    '''
    plt.title(
        f'{formalize_name(auc_type)} Auction for {N} Players',
        fontsize=14
    )

    plt.xlabel(
        'State (Value)',
        fontsize=14
    )

    plt.ylabel(
        'Action (Bid)',
        fontsize=14
    )

    plt.legend(
        loc='upper left',
        fontsize=12
    )

    axes = plt.gca()

    axes.set_xlim([
        0,
        max_revenue
    ])

    axes.set_ylim([
        0,
        max_revenue
    ])


def plot_expected_bid_curve(states: np.ndarray, auc_type: str, N: int, r: list, t: float,
                            max_revenue: float, count_zeros: int) -> None:
    '''
    Plots the theoretical (expected) bidding curve based on auction type and parameters.
    The function adds the expected bid curve to the current plot.

    Args:
        states (ndarray): The array of states (values) used in the auction.
        auc_type (str): The type of auction.
        N (int): The number of agents.
        r (list): Risk aversion coefficient for each agent.
        max_revenue (float): The maximum possible revenue.
        count_zeros (int): The number of agents with zero bids.
    '''

    def _plot(
        y_vals,
        label_suffix='',
        color='#AD1515',
        linestyle='-',
        linewidth=1.0,
        alpha=1.0
    ):
        plt.plot(
            states,
            y_vals,
            label=f'Expected bid {label_suffix}',
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha
        )

    asymmetric = not np.allclose(r, r[0])

    if asymmetric:
        expected_actions = calculate_asymmetric_expected_action(
            N,
            auc_type,
            states,
            r,
            t,
            max_revenue
        )

        colors = [
            '#AD1515',
            '#7B14AF',
            '#FF7F0E',
            '#2CA02C',
            '#17BECF',
            '#D62728'
        ]

        for k, expected in enumerate(expected_actions):
            _plot(
                expected,
                label_suffix=f'agent {k + 1}, r={r[k]}',
                color=colors[k % len(colors)]
            )

        return

    r_value = r[0]

    match auc_type:

        case 'first_price' | 'joint_first_price':
            _plot(
                states
                * (N - 1)
                / (N - 1 + r_value)
            )

        case 'second_price':
            _plot(states)

        case 'tariff_discount':
            _plot(
                (1 - (states / max_revenue))
                * (N - 1)
                / N
            )

        case 'common_value':
            _plot(states)

        case 'all_pay':
            expected = calculate_all_pay_optimal_bid(
                states,
                N,
                r_value
            )

            _plot(
                expected,
                label_suffix=f' N={N}, r={r_value}'
            )

            active_agents = (
                N - count_zeros
            )

            if 0 < active_agents < N:
                alt_exp = [
                    (s**active_agents)
                    * (active_agents - 1)
                    / active_agents
                    for s in states
                ]

                _plot(
                    alt_exp,
                    label_suffix=f' N={active_agents}',
                    color='#7B14AF',
                    linestyle='--',
                    linewidth=0.5,
                    alpha=0.5
                )

        case 'partial_all_pay':
            exponent = (
                1
                + t * (N - 1)
            )

            _plot(
                [
                    (s**exponent)
                    * (N - 1)
                    / N
                    for s in states
                ],
                label_suffix=f'N={N}, t={t}'
            )

            active_agents = (
                N - count_zeros
            )

            if 0 < active_agents < N:
                alt_exp = [
                    (s**active_agents)
                    * (active_agents - 1)
                    / active_agents
                    for s in states
                ]

                _plot(
                    alt_exp,
                    label_suffix=f'N={active_agents}',
                    color='#7B14AF',
                    linestyle='--',
                    linewidth=0.5,
                    alpha=0.5
                )


def manualTesting(agents: list, N: int, episode: int, n_episodes: int, auc_type: str = 'first_price',
                  r: list = [1], t: float = 1, max_revenue: float = 1, name: str = None) -> float:
    '''
    Performs manual testing of agent policies by plotting their bidding behavior against
    the theoretical benchmark and saving the resulting plot.

    Args:
        env:
        agents (list): List of agent objects.
        N (int): The number of agents.
        episode (int): The current episode.
        n_episodes (int): Total number of episodes.
        auc_type (str): The type of auction. Default is 'first_price'.
        r (list): A parameter used in specific auction types.
        max_revenue (float): The maximum possible revenue. Default is 1.

    Returns:
        avg_error (float): The average error between agent bids and expected bids.
    '''
    states, agents_actions, avg_error = calculate_agents_actions(
        agents,
        N,
        episode,
        auc_type,
        r,
        t,
        max_revenue
    )

    plt.close('all')

    plot_agents_actions(
        states,
        agents_actions
    )

    count_zeros = 0

    for i in agents_actions:
        if np.all(
            np.abs(i) <= 0.01
        ):
            count_zeros += 1

    plot_expected_bid_curve(
        states,
        auc_type,
        N,
        r,
        t,
        max_revenue,
        count_zeros
    )

    configure_plot_layout(
        auc_type,
        N,
        max_revenue
    )

    dir_path = (
        f'results/'
        f'{auc_type}/'
        f'N={N}/'
    )

    os.makedirs(
        dir_path,
        exist_ok=True
    )

    r_str = "_".join(
        (
            f"{int(r_i)}"
            if r_i == int(r_i)
            else f"{r_i}".replace('.', '_')
        )
        for r_i in r
    )

    fname = (
        f"{int(n_episodes / 1000)}k_"
        f"r{r_str}.png"
    )

    if name is not None:
        fname = (
            f"{int(n_episodes / 1000)}k_"
            f"r{r_str}_"
            f"{name}.png"
        )

    plt.savefig(
        f"{dir_path}{fname}"
    )

    return avg_error


def plot_errors(literature_error: list, loss_history: list, N: int, auction_type: str,
                n_episodes: int) -> None:
    '''
    Plots the literature error history and loss history over episodes and saves the resulting plots.

    Args:
        literature_error (list): A list containing the history of literature errors over episodes.
        loss_history (list): A list containing the history of loss values over episodes.
        N (int): The number of agents.
        auction_type (str): The type of auction.
        n_episodes (int): The total number of episodes, used for naming the saved plot files.

    Returns:
        None: The function saves the generated plots to files.
    '''
    dir_path = (
        f'results/'
        f'{auction_type}/'
        f'N={N}/'
    )

    os.makedirs(
        dir_path,
        exist_ok=True
    )

    plt.close('all')

    plt.plot(
        literature_error
    )

    plt.title(
        'Error history'
    )

    plt.xlabel(
        'Episode'
    )

    plt.ylabel(
        'Error'
    )

    plt.savefig(
        f'{dir_path}/'
        f'literature_error'
        f'{int(n_episodes/1000)}k.png'
    )

    plt.close('all')

    plt.plot(
        loss_history
    )

    plt.title(
        'Loss history'
    )

    plt.xlabel(
        'Episode'
    )

    plt.ylabel(
        'Loss'
    )

    plt.savefig(
        f'{dir_path}/'
        f'loss_history'
        f'{int(n_episodes/1000)}k.png'
    )