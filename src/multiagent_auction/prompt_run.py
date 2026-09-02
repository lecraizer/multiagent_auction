import os
import json
import subprocess
import requests
import time

from pathlib import Path


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free router. You can replace this with a specific free model later.
MODELS = [
    "cohere/north-mini-code:free",
    "z-ai/glm-5.2:free",
    "liquid/lfm-2.5-2.6b:free",
    "minimax/minimax-m2.7:free",
    "dots-studio/dots-3-note-preview:free",
    "openrouter/free",
]

PROMPT_PATH = Path(__file__).parent.parent.parent / "prompt.txt"

SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

VALID_AUCTIONS = {
    "first_price",
    "second_price",
    "all_pay",
    "partial_all_pay",
}


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("OpenRouter response must be a JSON object.")

    if "auction" in config:
        if not isinstance(config["auction"], str):
            raise ValueError(
                f"'auction' must be a string, got: {config['auction']}"
            )

        if config["auction"] not in VALID_AUCTIONS:
            raise ValueError(
                f"Unknown auction type: {config['auction']}"
            )

    if "target_auction" in config:
        if not isinstance(config["target_auction"], str):
            raise ValueError("'target_auction' must be a string.")

        if config["target_auction"] not in VALID_AUCTIONS:
            raise ValueError(
                f"Unknown target auction type: {config['target_auction']}"
            )

    integer_fields = {
        "episodes",
        "players",
        "extra_players",
        "batch",
    }

    for field in integer_fields:
        if field in config and not isinstance(config[field], int):
            raise ValueError(f"'{field}' must be an integer.")

    float_fields = {
        "noise",
        "all_pay_exponent",
        "ponderated",
        "gradient_floor",
    }

    for field in float_fields:
        if field in config and not isinstance(config[field], (int, float)):
            raise ValueError(f"'{field}' must be numeric.")

    bool_fields = {
        "trained",
        "transfer_learning",
        "save",
        "gif",
        "show_gui",
    }

    for field in bool_fields:
        if field in config and not isinstance(config[field], bool):
            raise ValueError(f"'{field}' must be boolean.")

    if "aversion_coef" in config:
        if not isinstance(config["aversion_coef"], list):
            raise ValueError("'aversion_coef' must be a list.")

        if not all(isinstance(x, (int, float)) for x in config["aversion_coef"]):
            raise ValueError(
                "All values in 'aversion_coef' must be numeric."
            )

    if "players" in config and "aversion_coef" in config:
        if len(config["aversion_coef"]) != config["players"]:
            raise ValueError(
                "The number of risk-aversion coefficients must match "
                "the number of players."
            )

    return config


def parse_description(description: str) -> dict:
    api_key = None
    key_path = Path(__file__).parent.parent.parent / "rooter_key.txt"
    if key_path.exists():
        api_key = key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("OpenRouter API key not found.")

    last_error = None

    for model in MODELS:
        print(f"\nTrying model: {model}")

        start_time = time.perf_counter()

        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": description,
                        },
                    ],
                    "temperature": 0,
                },
                timeout=30,
            )

            elapsed_time = time.perf_counter() - start_time

            # Model unavailable, rate limited, etc.
            if not response.ok:
                print(
                    f"Failed ({response.status_code}) "
                    f"after {elapsed_time:.2f}s"
                )
                last_error = response.text
                continue

            data = response.json()

            model_used = data.get("model", model)

            try:
                content = data["choices"][0]["message"]["content"]
                config = json.loads(content)
                config = validate_config(config)

            except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
                print(
                    f"Invalid response after {elapsed_time:.2f}s"
                )
                last_error = exc
                continue

            print(f"Model used: {model_used}")
            print(f"Response time: {elapsed_time:.2f} seconds")

            return config

        except requests.RequestException as exc:
            elapsed_time = time.perf_counter() - start_time

            print(
                f"Request failed after {elapsed_time:.2f}s"
            )

            last_error = exc
            continue

    raise RuntimeError(
        f"All models failed. Last error: {last_error}"
    )


def config_to_cli(config: dict) -> list:
    args = []

    mapping = {
        "auction": "-a",
        "target_auction": "-ta",
        "batch": "-b",
        "trained": "-d",
        "episodes": "-e",
        "gif": "-g",
        "players": "-n",
        "noise": "-z",
        "all_pay_exponent": "-t",
        "ponderated": "-p",
        "save": "-s",
        "transfer_learning": "-tl",
        "extra_players": "-x",
        "show_gui": "-sg",
        "gradient_floor": "-gf",
    }

    for key, flag in mapping.items():
        if key in config:
            value = config[key]

            if isinstance(value, bool):
                value = str(value).lower()

            args.extend([flag, str(value)])

    if "aversion_coef" in config:
        args.append("-r")
        args.extend(str(x) for x in config["aversion_coef"])

    return args


def main():
    print("Describe the auction you want to simulate:")
    description = input("> ").strip()

    if not description:
        print("No description provided.")
        return

    config = parse_description(description)

    print("\nInterpreted configuration:")
    print(json.dumps(config, indent=2))

    answer = input("\nRun simulation? [Y/n] ").strip().lower()

    if answer == "n":
        return

    cli_args = config_to_cli(config)

    command = [
        "python",
        "src/multiagent_auction/run.py",
        *cli_args,
    ]

    print("\nRunning:")
    print(" ".join(command))
    print()

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()