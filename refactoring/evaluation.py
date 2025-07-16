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
    tuples_list = list(zip(*others_bids))
    max_bids = [max(t) for t in tuples_list]
    win_prob = sum(own_bid > b for b in max_bids) / len(max_bids)
    if win_prob == 0.0: return 0.0
    if auc_type == 'second_price':
        bids_below_own_bid = [b for b in max_bids if b < own_bid]
        if not bids_below_own_bid: return 0.0
        expected_second_bid = np.mean(bids_below_own_bid)
        return (own_value - expected_second_bid) * win_prob
    return (own_value - own_bid) * win_prob

def get_all_bids_except(agents, k, n_bids):
    others_bids = []
    for j in range(len(agents)):
        if j != k:
            bids = [agents[j].choose_action(np.random.random(), 0, evaluation=True)[0] 
                    for _ in range(n_bids)]
            others_bids.append(bids)
    return others_bids
    
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

    for k in range(len(agents)):
        others_bids = get_all_bids_except(agents, k, n_bids)
        revenue_diffs = []
        for _ in range(200):
            own_value = np.random.random()       
            opt_empirical_revenue = 0.0
            optimal_bid = 0.0
            for b in range(grid_precision):
                own_bid = b/grid_precision
                revenue = get_empirical_revenue(own_value, own_bid, others_bids, auc_type)
                if revenue > opt_empirical_revenue:
                    opt_empirical_revenue = revenue
                    optimal_bid = own_bid

            own_optimal_bid = agents[k].choose_action(own_value, 0, evaluation=True)[0]
            win_prob = sum(own_optimal_bid > b for b in [max(t) for t in zip(*others_bids)]) / len(others_bids[0])
            opt_player_revenue = (own_value - own_optimal_bid)*win_prob
            revenue_diffs.append(abs(opt_empirical_revenue - opt_player_revenue))

        avg_revenue_diff = np.mean(revenue_diffs)
        revenue_diffs_all.append(avg_revenue_diff)
        print('\nAvg diff bids player', k+1, ':', avg_revenue_diff)
    print('\nAverage difference all players:', np.mean(revenue_diffs_all))