import argparse
import json

def parse_args():
    """
    Parse command line arguments + load technical parameters from params.json
    """
    parser = argparse.ArgumentParser(description='Parse terminal input information')

    # Only keep essential arguments for user
    parser.add_argument('-a', '--auction', type=str, default='first_price', help='Auction type')
    parser.add_argument('-d', '--trained', type=bool, default=0, help='Load models instead of training them (0 or 1)')
    parser.add_argument('-e', '--episodes', type=int, default=2000, help='Total number of training episodes')
    parser.add_argument('-n', '--players', type=int, default=2, help='Total number of players')
    parser.add_argument('-r', '--aversion_coef', type=float, default=1.0, help='Aversion coefficient')
    parser.add_argument('-t', '--transfer_learning', type=bool, default=0, help='Use transfer learning (0 or 1)')
    parser.add_argument('-x', '--extra_players', type=int, default=0, help='Extra players')

    args = parser.parse_args()

    # Load technical parameters from JSON
    with open("params.json", "r") as f:
        config = json.load(f)

    # Combine and separate CLI args and config
    cli_args = vars(args)

    print('\n--- Configuration ---\n')
    max_len = max(len(k.replace('_', ' ')) for k in cli_args)

    for k, v in cli_args.items():
        name = k.replace('_', ' ').title()

        # Interpret 0 or 1 as booleans if that makes sense
        if isinstance(v, str):
            v = v.replace('_', ' ').title()
        elif isinstance(v, int) and v in (0, 1):
            v = 'Yes' if v == 1 else 'No'

        print(f'{name:<{max_len}} : {v}')

    print('\n-------------------------------\n')

    return (
        args.auction,
        config["batch"],
        args.trained,
        args.episodes,
        config["gif"],
        args.players,
        config["noise"],
        config["ponderated"],
        args.aversion_coef,
        config["save"],
        config["alert"],
        args.transfer_learning,
        args.extra_players,
        config["executions"]
    )
