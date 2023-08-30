import numpy as np

def evaluate(agents, N):
    '''
    Evaluate the agents by comparing their bids to the optimal bids
    '''
    # list of bids for player 1 given random private values
    bids_player1 = [agents[0].choose_action(np.random.random(), 0, evaluation=True)[0] for i in range(N)]

    differences2 = []
    for i in range(100):
        v2 = np.random.random()
        # print('v2:', v2)
        
        # Left revenue and bid
        opt_left_revenue = 0.0
        opt_left_bid = 0.0
        for b in range(N):
            b2 = b/N
            win_prob = sum([b2 > b1 for b1 in bids_player1])/N
            left_revenue = (v2 - b2) * win_prob
            if left_revenue > opt_left_revenue:
                opt_left_revenue = left_revenue
                opt_left_bid = b2

        # print('Optimal left bid:', opt_left_bid)
        # print('Optimal left revenue:', opt_left_revenue)

        # Right revenue and bid
        b2 = agents[1].choose_action(v2, 0, evaluation=True)[0]
        win_prob = sum([b2 > b1 for b1 in bids_player1])/N
        opt_right_revenue = (v2 - b2)*win_prob

        # print('Optimal right bid:', b2)
        # print('Optimal right revenue:', opt_right_revenue)

        # Absolute difference between optimal revenues
        diff = abs(opt_left_revenue - opt_right_revenue)
        # print('Difference:', diff)
        differences2.append(diff)

    print('Average difference player 2:', np.mean(differences2))

    # do the exact same thing for player 2

    # list of bids for player 2 given random private values
    bids_player2 = [agents[1].choose_action(np.random.random(), 0, evaluation=True)[0] for i in range(N)]

    differences1 = []
    for i in range(100):
        v1 = np.random.random()
        # print('v1:', v1)
        
        # Left revenue and bid
        b1 = agents[0].choose_action(v1, 0, evaluation=True)[0]
        win_prob = sum([b1 > b2 for b2 in bids_player2])/N
        opt_left_revenue = (v1 - b1)*win_prob

        # print('Optimal left bid:', b1)
        # print('Optimal left revenue:', opt_left_revenue)

        # Right revenue and bid
        opt_right_revenue = 0.0
        opt_right_bid = 0.0
        for b in range(N):
            b1 = b/N
            win_prob = sum([b1 > b2 for b2 in bids_player2])/N
            right_revenue = (v1 - b1) * win_prob
            if right_revenue > opt_right_revenue:
                opt_right_revenue = right_revenue
                opt_right_bid = b1
        # print('Optimal right bid:', opt_right_bid)
        # print('Optimal right revenue:', opt_right_revenue)

        # Absolute difference between optimal revenues
        diff = abs(opt_left_revenue - opt_right_revenue)
        # print('Difference:', diff)
        differences1.append(diff)

    print('Average difference player 2:', np.mean(differences1))

    return np.mean(differences1), np.mean(differences2)





        