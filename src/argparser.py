import argparse

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Parse terminal input information')

    parser.add_argument('-a', '--auction', type=str, default='first_price', help='Auction type')
    parser.add_argument('-b', '--batch', type=int, default=64, help='Batch size')
    parser.add_argument('-d', '--trained', type=int, default=0, help='Load models instead of training them (0 or 1)')
    parser.add_argument('-e', '--episodes', type=int, default=2000, help='Total number of training episodes')
    parser.add_argument('-g', '--gif', type=int, default=0, help='Create state-actions plot gif (0 or 1)')
    parser.add_argument('-n', '--players', type=int, default=2, help='Total number of players')
    parser.add_argument('-o', '--noise', type=float, default=0.2, help='Noise standard deviation')
    parser.add_argument('-p', '--ponderated', type=int, default=100, help='Ponderated average size')
    parser.add_argument('-r', '--aversion_coef', type=float, default=1.0, help='Aversion coefficient')
    parser.add_argument('-s', '--save', type=int, default=0, help='Save plot (0 or 1)')
    parser.add_argument('-t', '--alert', type=int, default=0, help='Beep when training is done (0 or 1)')
    parser.add_argument('-tl', '--transfer_learning', type=int, default=0, help='Use transfer learning (0 or 1)')
    parser.add_argument('-x', '--extra_players', type=int, default=0, help='Extra players')
    parser.add_argument('-z', '--executions', type=int, default=1, help='Number of executions')

    args = parser.parse_args()

    # Print arguments in a clean way
    print('\n--- Configuration ---')
    for k, v in vars(args).items():
        print(f'{k.replace("_", " ").capitalize()}: {v}')
    print('---------------------\n')

    return (
        args.auction,
        args.batch,
        args.trained,
        args.episodes,
        args.gif,
        args.players,
        args.noise,
        args.ponderated,
        args.aversion_coef,
        args.save,
        args.alert,
        args.transfer_learning,
        args.extra_players,
        args.executions,
    )
