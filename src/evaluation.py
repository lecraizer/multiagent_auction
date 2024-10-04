# Module to evaluate the performance of the agents and check if agents converge to their optimal bids

import numpy as np


def evaluate(agents, n_bids=100):
    '''
    Evaluate the agents by comparing their bids to the optimal bids
    '''
    # list of bids for player 1 given random private values
    bids_player1 = [agents[0].choose_action(np.random.random(), 0, evaluation=True)[0] for i in range(n_bids)]

    differences = []
    sum_opt_revenues = 0.0
    for i in range(n_bids):
        own_value = np.random.random()
        # print('own_value:', own_value)
        
        # Left revenue and bid
        opt_empirical_revenu = 0.0
        # opt_left_bid = 0.0
        for b in range(n_bids): # grid search
            b2 = b/n_bids
            win_prob = sum([b2 > b1 for b1 in bids_player1])/n_bids # empirical win probability
            empirical_revenu = (own_value - b2) * win_prob
            if empirical_revenu > opt_empirical_revenu:
                opt_empirical_revenu = empirical_revenu
                # opt_left_bid = b2
        sum_opt_revenues += opt_empirical_revenu

        # print('Optimal left bid:', opt_left_bid)
        # print('Optimal left revenue:', opt_empirical_revenu)

        # Right revenue and bid
        b2 = agents[1].choose_action(own_value, 0, evaluation=True)[0]
        win_prob = sum([b2 > b1 for b1 in bids_player1])/n_bids
        opt_right_revenue = (own_value - b2)*win_prob

        # print('Optimal right bid:', b2)
        # print('Optimal right revenue:', opt_right_revenue)

        # Absolute difference between optimal revenues
        diff = abs(opt_empirical_revenu - opt_right_revenue)
        # print('Difference:', diff)
        differences.append(diff)

    print('Average difference player 2:', np.mean(differences))

    equation = np.mean(differences) / (sum_opt_revenues/n_bids)
    print('\nError equation Bichler:', equation)


def calculate_win_probability(own_bid, others_bids):
    '''
    Calculate the empirical win probability of player given the others' bids
    '''
    win_prob = 1
    # Iterate over each list of bids
    for bids in others_bids:
        win_prob *= sum(own_bid > b for b in bids) / len(bids)
    return win_prob


def evaluate_N(agents, n_bids=100, grid_precision=100, auc_type='first_price'):
    '''
    Evaluate the agents by comparing their bids to the optimal bids
    '''

    # for i in range(n_iter=1000000):

    sum_all_players = 0.0
    for k in range(len(agents)):
        others_bids = []
        for j in range(len(agents)):
            if j != k:
                bids = [agents[j].choose_action(np.random.random(), 0, evaluation=True)[0] for i in range(n_bids)]
                others_bids.append(bids)

        differences = []
        diff_bids_list = []
        sum_opt_revenues = 0.0
        for i in range(n_bids):
            own_value = np.random.random() # random private value        
            opt_empirical_revenue = 0.0 # optimal left revenue
            optimal_bid = 0.0
            for b in range(grid_precision): # grid search
                own_bid = b/grid_precision # own bid
                win_prob = calculate_win_probability(own_bid, others_bids) # empirical win probability
                empirical_revenue = (own_value - own_bid) * win_prob # empirical revenue
                if auc_type == 'second_price':
                    expected_second_bid = (len(agents)-1)/len(agents)
                    empirical_revenue = (own_value - expected_second_bid * own_bid) * win_prob
                if empirical_revenue > opt_empirical_revenue: # update optimal revenue
                    opt_empirical_revenue = empirical_revenue
                    optimal_bid = own_bid
            sum_opt_revenues += opt_empirical_revenue

            # Player revenue and bid
            own_optimal_bid = agents[k].choose_action(own_value, 0, evaluation=True)[0]
            win_prob = calculate_win_probability(own_optimal_bid, others_bids)
            opt_player_revenue = (own_value - own_optimal_bid)*win_prob

            diff_bids = abs(own_optimal_bid - optimal_bid)
            diff_bids_list.append(diff_bids)

            # Absolute difference between optimal revenues
            diff = abs(opt_empirical_revenue - opt_player_revenue)
            # diff = abs(opt_empirical_revenue - opt_player_revenue) / opt_empirical_revenue
            differences.append(diff)

        # print('\nAverage difference player', k+1, ':', np.mean(differences))
        sum_all_players += np.mean(differences)

        print('\nAvg diff bids player', k+1, ':', np.mean(diff_bids_list))

    # print('\nAverage difference all players:', sum_all_players/len(agents))

    # equation = np.mean(differences) / (sum_opt_revenues/sample_size)
    # print('\nError equation Bichler:', equation)


def evaluate_one_agent(agents, k=0, num_samples=100):
    bids_other_players = []
    for j in range(len(agents)):
        if j != k:
            bids = [agents[j].choose_action(np.random.random(), 0, evaluation=True)[0] for i in range(num_samples)]
            bids_other_players.append(bids)
    
    soma = 0.0
    for z in range(100):
        n_grid=200
        opt_empirical_revenue = 0.0
        best_bid = 0.0
        own_value = np.random.random()
        for b in range(n_grid): # grid search
            own_bid = b/n_grid # own bid
            win_prob = 1
            for other_bids in bids_other_players:
                win_prob *= sum(own_bid > b for b in other_bids) / num_samples 
            empirical_revenue = (own_value - own_bid) * win_prob # empirical revenue
            if empirical_revenue > opt_empirical_revenue:
                opt_empirical_revenue = empirical_revenue
                best_bid = own_bid

        own_optimal_bid = agents[k].choose_action(own_value, 0, evaluation=True)[0]
        diff = abs(best_bid - own_optimal_bid)
        soma += diff
    
    print('Average difference:', soma/100)
    

def evaluate_one_agent_against_itself(agents, k=0, num_samples=100):

    # num_samples = 1000
    bids_player = [[agents[k].choose_action(np.random.random(), 0, evaluation=True)[0] for i in range(num_samples)] for j in range(len(agents)-1)]

    aux = 0.9
    
    count = 0
    for bid in bids_player[0]:
        if bid < aux:
            count += 1

    # print('\nTotal number of bids:', num_samples)
    # print('Percentage of bids below', aux, ':', count/num_samples)    
    
    # quit()
    N = len(agents)
    soma = 0.0
    for z in range(100):
        n_grid=100
        opt_empirical_revenue = 0.0
        best_bid = 0.0
        own_value = np.random.random()
        for b in range(n_grid): # grid search
            own_bid = b/n_grid # own bid
            win_prob = 1
            for other_bids in bids_player:
                win_prob *= sum(own_bid > b for b in other_bids) / num_samples
            # empirical revenue 2nd price auction
            empirical_revenue = (own_value - (N-1)/N * own_bid) * win_prob
            if empirical_revenue > opt_empirical_revenue:
                opt_empirical_revenue = empirical_revenue
                best_bid = own_bid

            # if abs(own_bid - own_value) < 0.01:
            #     print('\n===================')
            #     print('Own value:', own_value)
            #     print('Bid:', own_bid)
            #     print('Revenue:', empirical_revenue)
                    
        # print('\nValue:', own_value)
        # print('Best bid:', best_bid)
        # print('Optimal revenue:', opt_empirical_revenue)

        own_optimal_bid = agents[k].choose_action(own_value, 0, evaluation=True)[0]
        diff = abs(best_bid - own_optimal_bid)
        soma += diff
    
    print('Average difference:', soma/100)


def get_empirical_revenue(own_value, own_bid, others_bids, auc_type='first_price'):
    '''
    '''
    # zip the lists of bids and then create a single list (1d list) with the maximum bid of each pair
    tuples_list = list(zip(*others_bids))
    max_bids = [max(t) for t in tuples_list]

    # calculate the win probability
    win_prob = sum(own_bid > b for b in max_bids) / len(max_bids)
    if win_prob == 0.0:
        return 0.0
    # get expected empirical revenue
    if auc_type == 'second_price':
        bids_below_own_bid = [b for b in max_bids if b < own_bid]
        expected_second_bid = sum(bids_below_own_bid)/len(bids_below_own_bid)
        return (own_value - expected_second_bid) * win_prob
    
    return (own_value - own_bid) * win_prob


def new_evaluate_agents(agents, n_bids=100, grid_precision=100, auc_type='first_price'):
    '''
    Evaluate the agents by comparing their bids to the optimal bids
    '''
    sums_of_diffs = []
    for k in range(len(agents)):
        others_bids = []
        for j in range(len(agents)):
            if j != k:
                bids = [agents[j].choose_action(np.random.random(), 0, evaluation=True)[0] for i in range(n_bids)]
                others_bids.append(bids)

        differences = []
        diff_bids_list = []
        sum_opt_revenues = 0.0
        for i in range(200):
            own_value = np.random.random() # random private value        
            opt_empirical_revenue = 0.0 # optimal left revenue
            optimal_bid = 0.0
            for b in range(grid_precision): # grid search
                own_bid = b/grid_precision # own bid
                # empirical_revenue = (own_value - own_bid) * win_prob # empirical revenue
                # win_prob = calculate_win_probability(own_bid, others_bids)
                # expected_second_bid = (len(agents)-1)/len(agents)
                # empirical_revenue = (own_value - expected_second_bid * own_bid) * win_prob
                empirical_revenue = get_empirical_revenue(own_value, own_bid, others_bids, auc_type)
                if empirical_revenue > opt_empirical_revenue: # update optimal revenue
                    opt_empirical_revenue = empirical_revenue
                    optimal_bid = own_bid
            sum_opt_revenues += opt_empirical_revenue

            # Player revenue and bid
            own_optimal_bid = agents[k].choose_action(own_value, 0, evaluation=True)[0]
            win_prob = get_empirical_revenue(own_value, own_optimal_bid, others_bids, auc_type)
            opt_player_revenue = (own_value - own_optimal_bid)*win_prob

            diff_bids = abs(own_optimal_bid - optimal_bid)
            diff_bids_list.append(diff_bids)

            # Absolute difference between optimal revenues
            diff = abs(opt_empirical_revenue - opt_player_revenue)
            differences.append(diff)

        sums_of_diffs.append(np.mean(diff_bids_list))

        print('\nAvg diff bids player', k+1, ':', np.mean(diff_bids_list))

    print('\nAverage difference all players:', np.mean(sums_of_diffs))