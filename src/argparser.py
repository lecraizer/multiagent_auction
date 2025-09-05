import json

def load_args() -> tuple:
    """
    Load parameters from json file.
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
        config["ponderated"],
        config["aversion_coef"],
        config["save"],
        config["transfer_learning"],
        config["extra_players"],
        config["show_gui"]
    )