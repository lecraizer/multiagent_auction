import numpy as np

def get_empirical_revenue(own_value, own_bid, others_bids, auc_type='first_price'):
    tuples_list = list(zip(*others_bids))
    max_bids = [max(t) for t in tuples_list]
    win_prob = sum(own_bid > b for b in max_bids) / len(max_bids)
    if win_prob == 0.0: return 0.0
    if auc_type == 'second_price':
        bids_below_own_bid = [b for b in max_bids if b < own_bid]
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
    sums_of_diffs = []
    for k in range(len(agents)):
        others_bids = get_all_bids_except(agents, k, n_bids)
        differences = []
        diff_bids_list = []
        sum_opt_revenues = 0.0
        for _ in range(200):
            own_value = np.random.random()       
            opt_empirical_revenue = 0.0
            optimal_bid = 0.0
            for b in range(grid_precision):
                own_bid = b/grid_precision
                empirical_revenue = get_empirical_revenue(own_value, own_bid, others_bids, auc_type)
                if empirical_revenue > opt_empirical_revenue:
                    opt_empirical_revenue = empirical_revenue
                    optimal_bid = own_bid
            sum_opt_revenues += opt_empirical_revenue

            own_optimal_bid = agents[k].choose_action(own_value, 0, evaluation=True)[0]
            win_prob = get_empirical_revenue(own_value, own_optimal_bid, others_bids, auc_type)
            opt_player_revenue = (own_value - own_optimal_bid)*win_prob
            diff_bids_list.append(abs(own_optimal_bid - optimal_bid))

            differences.append(abs(opt_empirical_revenue - opt_player_revenue))
        sums_of_diffs.append(np.mean(diff_bids_list))
        print('\nAvg diff bids player', k+1, ':', np.mean(diff_bids_list))
    print('\nAverage difference all players:', np.mean(sums_of_diffs))