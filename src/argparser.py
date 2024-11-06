import argparse

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Parse terminal input information')

    n_episodes = 2000 # number of episodes
    N = 2 # number of players
    noise_std = 0.2 # noise standard deviation
    r = 1 # aversion coefficient
    BS = 64 # batch size
    ponderated_avg = 100 # ponderated average size
    auction = 'first_price' # auction type
    z = 1 # number of executions
    save_plot = 1 # save plot
    alert = 0 # alert
    trained = 0 # set to '1' to load models instead of training them
    create_gif = 0 # create gif
    extra_players = 2 # extra players
    transfer_learning = False # transfer learning

    parser.add_argument('-a', '--auction', type=str, help='Auction type')
    parser.add_argument('-b', '--batch', type=int, help='Batch size')
    parser.add_argument('-d', '--trained', type=int, help='Load models instead of training them')
    parser.add_argument('-e', '--episodes', type=int, help='Total number of training episodes')
    parser.add_argument('-g', '--gif', type=int, help='Create state-actions plot gif')
    parser.add_argument('-n', '--players', type=int, help='Total number of players')
    parser.add_argument('-o', '--noise', type=float, help='Noise standard deviation')
    parser.add_argument('-p', '--ponderated', type=int, help='Ponderated average size')
    parser.add_argument('-r', '--aversion_coef', type=float, help='Aversion coefficient')
    parser.add_argument('-s', '--save', type=int, help='Save plot')
    parser.add_argument('-t', '--alert', type=int, help='Beep when training is done')
    parser.add_argument('-tl', '--transfer_learning', type=bool, help='Transfer learning')
    parser.add_argument('-x', '--extra_players', type=int, help='Extra players')
    parser.add_argument('-z', '--executions', type=int, help='Number of executions')
    args = parser.parse_args()

    # if arguments are passed, overwrite default values
    if args.auction: # a
        auction = args.auction
    if args.batch: # b
        BS = args.batch
    if args.trained: # d
        trained = args.trained
    if args.episodes: # e
        n_episodes = args.episodes
    if args.gif: # g
        create_gif = args.gif
    if args.players: # n
        N = args.players
    if args.noise: # o
        noise_std = args.noise
    if args.ponderated: # p
        ponderated_avg = args.ponderated
    if args.aversion_coef: # r
        r = args.aversion_coef
    if args.save: # s
        save_plot = args.save
    if args.alert: # t
        alert = args.alert
    if args.transfer_learning: # tl
        transfer_learning = args.transfer_learning
    if args.extra_players: # x
        extra_players = args.extra_players
    if args.executions: # z
        z = args.executions

    print('Auction type: ', auction) # a
    print('Batch size: ', BS) # b
    print('Load models: ', trained) # d
    print('Number of episodes: ', n_episodes) # e
    print('Create gif: ', create_gif) # g
    print('Number of players: ', N) # n
    print('Noise standard deviation: ', noise_std) # o
    print('Ponderated average size: ', ponderated_avg) # p
    print('Aversion coefficient: ', r) # r
    print('Save plot: ', save_plot) # s
    print('Alert: ', alert) # t
    print('Transfer learning: ', transfer_learning) # tl
    print('Extra players: ', extra_players) # x
    print('Number of executions: ', z) # z
    print('\n')

    return auction, BS, trained, n_episodes, create_gif, N, noise_std, ponderated_avg, r, save_plot, alert, transfer_learning, extra_players, z