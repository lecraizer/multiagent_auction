# Module to evaluate the performance of the agents and check if agents converge to their optimal bids

import numpy as np


def get_empirical_revenue(own_value, own_bid, others_bids, auc_type='first_price'):
    """
    Estimate the expected revenue for a given bid based on sampled opponent bids.

    Args:
        own_value (float): Agent's private value.
        own_bid (float): Agent's chosen bid.
        others_bids (list of lists): List of bid samples from other agents.
        auc_type (str): Auction type.

    Returns:
        float: Estimated revenue.
    """
    # Combine bids at each round into tuples and get max bid from each round
    tuples_list = list(zip(*others_bids))
    max_bids = [max(t) for t in tuples_list]

    # Compute empirical win probability (frequency of winning)
    win_prob = sum(own_bid > b for b in max_bids) / len(max_bids)
    if win_prob == 0.0:
        return 0.0

    # Estimate revenue depending on auction type
    if auc_type == 'second_price':
        bids_below_own_bid = [b for b in max_bids if b < own_bid]
        if not bids_below_own_bid:
            return 0.0
        expected_second_bid = sum(bids_below_own_bid) / len(bids_below_own_bid)
        return (own_value - expected_second_bid) * win_prob

    return (own_value - own_bid) * win_prob


def evaluate_agents(agents, n_bids=100, grid_precision=100, auc_type='first_price'):
    """
    Evaluate the trained agents by comparing their bidding strategy to the optimal empirical strategy.

    Args:
        agents (list): List of trained agents.
        n_bids (int): Number of samples to estimate other agents' bid distributions.
        grid_precision (int): Number of points in bid grid search.
        auc_type (str): Auction type.

    Prints:
        Table with average revenue differences for each player and overall.
    """
    revenue_diffs_all = []

    print('\nEvaluation Results (based on revenue difference)')
    print('-----------------------------------------------')
    print(f'{"Player":<20}{"Avg Revenue Diff":>20}')
    print('-----------------------------------------------')

    for k in range(len(agents)):
        # Estimate empirical bid distributions of other agents
        others_bids = []
        for j in range(len(agents)):
            if j != k:
                bids = [agents[j].choose_action(np.random.random(), 0, evaluation=True)[0] for _ in range(n_bids)]
                others_bids.append(bids)

        revenue_diffs = []
        # bid_diffs = []  # Optional: uncomment to compare bids

        for _ in range(200):
            own_value = np.random.random()

            # Grid search for optimal empirical bid
            opt_empirical_revenue = 0.0
            optimal_bid = 0.0
            for b in range(grid_precision):
                own_bid = b / grid_precision
                revenue = get_empirical_revenue(own_value, own_bid, others_bids, auc_type)
                if revenue > opt_empirical_revenue:
                    opt_empirical_revenue = revenue
                    optimal_bid = own_bid

            # Agent's bid and revenue
            own_optimal_bid = agents[k].choose_action(own_value, 0, evaluation=True)[0]
            win_prob = sum(own_optimal_bid > b for b in [max(t) for t in zip(*others_bids)]) / len(others_bids[0])
            opt_player_revenue = (own_value - own_optimal_bid) * win_prob
            revenue_diff = abs(opt_empirical_revenue - opt_player_revenue)
            revenue_diffs.append(revenue_diff)

            # Optional: difference in bid values
            # bid_diff = abs(optimal_bid - own_optimal_bid)
            # bid_diffs.append(bid_diff)

        avg_revenue_diff = np.mean(revenue_diffs)
        revenue_diffs_all.append(avg_revenue_diff)
        print(f'{"Player " + str(k+1):<20}{avg_revenue_diff:>20.6f}')

    print('-----------------------------------------------')
    print(f'{"All Players Average":<20}{np.mean(revenue_diffs_all):>20.6f}\n')
