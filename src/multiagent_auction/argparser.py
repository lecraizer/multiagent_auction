import json
import argparse
from logging import config


def str_to_bool(value):
    """
    Converts a command-line string to boolean.
    """
    if isinstance(value, bool):
        return value

    if value.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif value.lower() in ('false', '0', 'no', 'n'):
        return False

    raise argparse.ArgumentTypeError('Boolean value expected.')


def load_args() -> tuple:
    """
    Load simulation parameters from the `params.json` configuration file.
    Command-line arguments override the values defined in the configuration file.
    The parameters are returned in a fixed tuple order.

    Returns:
        tuple: A tuple containing the following configuration values in order:
        - auction (str): The auction mechanism to be used.
        - target_auction (str): The target auction for transfer learning.
        - batch (int): Batch size for training.
        - trained (bool): Whether to load a pre-trained model.
        - episodes (int): Number of training episodes.
        - gif (bool): Whether to generate a GIF of the simulation.
        - players (int): Number of participating players.
        - noise (float): Standard deviation of the action noise.
        - all_pay_exponent (float): Exponent used in partial all-pay auctions.
        - ponderated (float): Parameter used for ponderated evaluation.
        - aversion_coef (list): Risk aversion coefficient for each player.
        - save (bool): Whether to save plots or trained models.
        - transfer_learning (bool): Whether to enable transfer learning.
        - extra_players (int): Number of additional players.
        - show_gui (bool): Whether to display the graphical interface.
    """

    with open("params.json", "r") as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(
        description="Multi-Agent Auction Simulation"
    )

    parser.add_argument("-a", "--auction", type=str)
    parser.add_argument("-ta", "--target-auction", type=str)
    parser.add_argument("-b", "--batch", type=int)
    parser.add_argument("-d", "--trained", type=str_to_bool)
    parser.add_argument("-e", "--episodes", type=int)
    parser.add_argument("-f", "--gradient-floor", type=float)
    parser.add_argument("-n", "--players", type=int)
    parser.add_argument("-z", "--noise", type=float)
    parser.add_argument("-t", "--all-pay-exponent", type=float)
    parser.add_argument("-p", "--ponderated", type=float)
    parser.add_argument("-r","--aversion-coef", type=float, nargs="+")
    parser.add_argument("-s", "--save", type=str_to_bool)
    parser.add_argument("-tl", "--transfer-learning", type=str_to_bool)
    parser.add_argument("-x", "--extra-players", type=int)

    args = parser.parse_args()

    if args.auction is not None:
        config["auction"] = args.auction

    if args.target_auction is not None:
        config["target_auction"] = args.target_auction

    if args.batch is not None:
        config["batch"] = args.batch

    if args.trained is not None:
        config["trained"] = args.trained

    if args.episodes is not None:
        config["episodes"] = args.episodes

    if args.gradient_floor is not None:
        config["gradient_floor"] = args.gradient_floor

    if args.players is not None:
        config["players"] = args.players

    if args.noise is not None:
        config["noise"] = args.noise

    if args.all_pay_exponent is not None:
        config["all_pay_exponent"] = args.all_pay_exponent

    if args.ponderated is not None:
        config["ponderated"] = args.ponderated

    if args.aversion_coef is not None:
        config["aversion_coef"] = args.aversion_coef

    if config["aversion_coef"] is None:
        config["aversion_coef"] = [1.0] * config["players"]

    if args.save is not None:
        config["save"] = args.save

    if args.transfer_learning is not None:
        config["transfer_learning"] = args.transfer_learning

    if args.extra_players is not None:
        config["extra_players"] = args.extra_players

    return (
        config["auction"],
        config["target_auction"],
        config["batch"],
        config["trained"],
        config["episodes"],
        config["gradient_floor"],
        config["players"],
        config["noise"],
        config["all_pay_exponent"],
        config["ponderated"],
        config["aversion_coef"],
        config["save"],
        config["transfer_learning"],
        config["extra_players"],
    )