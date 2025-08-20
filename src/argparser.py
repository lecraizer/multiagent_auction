import json

def load_args():
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
        config["alert"],
        config["transfer_learning"],
        config["extra_players"],
        config["executions"],
        config["show_gui"]
    )