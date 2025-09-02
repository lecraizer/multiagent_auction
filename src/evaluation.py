import numpy as np

def get_empirical_revenue(env, own_value: float, own_bid: float, others_bids: list) -> float:
    """
    Estimate the expected revenue for a given bid based on sampled opponent bids.

    Args:
        env:
        own_value (float): Agent's private value.
        own_bid (float): Agent's chosen bid.
        others_bids (list): List of bid samples from other agents.

    Returns:
        float: Estimated revenue.
    """
    max_bids = [max(t) for t in list(zip(*others_bids))]
    win_prob = sum(own_bid > b for b in max_bids) / len(max_bids)
    if win_prob == 0.0: return 0.0
    value_paid = env.value_paid(own_bid, max_bids)
    return (own_value - value_paid) * win_prob

def get_all_bids_except(env, agents: list, k: int, n_bids: int) -> list:
    """
    Generate bid samples from all agents except one.

    Args:
        env:
        agents (list): List of agent objects.
        k (int): Index of the agent to exclude.
        n_bids (int): Number of bid samples to generate per agent.

    Returns:
        list: A list where each element contains `n_bids` sampled bids 
              from one of the other agents.
    """
    others_bids = []
    for j in range(len(agents)):
        if j != k:
            bids = [agents[j].choose_action(env.reset()[j], 0, evaluation=True)[0] 
                    for _ in range(n_bids)]
            others_bids.append(bids)
    return others_bids
    
def evaluate_agents(env, agents: list, n_bids: int=100, grid_precision: int=100, 
                    auc_type: str='first_price') -> None:
    """
    Evaluate the trained agents by comparing their bidding strategy to the optimal 
    empirical strategy.

    Args:
        env:
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
        others_bids = get_all_bids_except(env, agents, k, n_bids)
        revenue_diffs = []
        for _ in range(200):
            own_value = env.reset()[k]      
            opt_empirical_revenue = 0.0
            for b in range(grid_precision):
                own_bid = b/grid_precision
                revenue = get_empirical_revenue(env, own_value, own_bid, others_bids)
                if revenue > opt_empirical_revenue:
                    opt_empirical_revenue = revenue

            own_optimal_bid = agents[k].choose_action(own_value, 0, evaluation=True)[0]
            win_prob = sum(own_optimal_bid > b for b in [max(t) for t in zip(*others_bids)]
                           ) / len(others_bids[0])
            opt_player_revenue = (own_value - own_optimal_bid)*win_prob
            revenue_diffs.append(abs(opt_empirical_revenue - opt_player_revenue))

        avg_revenue_diff = np.mean(revenue_diffs)
        revenue_diffs_all.append(avg_revenue_diff)
        print('\nAvg diff bids player', k+1, ':', avg_revenue_diff)
    print('\nAverage difference all players:', np.mean(revenue_diffs_all))