import json

def load_args() -> tuple:
    """
    Load simulation parameters from the `params.json` configuration file.
    The function reads a JSON file located in the current working directory 
    and extracts the parameters required to run an auction simulation. 
    The parameters are returned in a fixed tuple order.

    Returns:
        tuple: A tuple containing the following configuration values in order:
        - auction (str): The auction mechanism to be used (e.g., "first_price", "second_price").
        - target_auction (str): The target auction for transfer learning.
        - batch (int): Batch size for training.
        - trained (bool): Whether to load a pre-trained model instead of starting fresh.
        - gif (bool): Whether to generate a GIF of the simulation.
        - players (int): Number of participating players in the auction.
        - noise (float): Standard deviation of the noise applied to bidding strategies.
        - all_pay_exponent (float): Exponent factor used in all-pay auction variants.
        - aversion_coef (float): Risk aversion coefficient for player strategies.
        - save (bool): Whether to save plots or trained models.
        - transfer_learning (bool): Whether to enable transfer learning between auctions.
        - extra_players (int): Number of additional players beyond the default setup.
        - show_gui (bool): Whether to display a graphical user interface during simulation.

    """
    with open("params.json", "r") as f:
        config = json.load(f)

    return (
        config["auction"],
        config["target_auction"],
        config["batch"],
        config["trained"],
        config["episodes"],
        config["gif"],
        config["players"],
        config["noise"],
        config["all_pay_exponent"],
        config["aversion_coef"],
        config["save"],
        config["transfer_learning"],
        config["extra_players"],
        config["show_gui"]
    )