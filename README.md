# Deep Reinforcement Learning in Auction Theory 
Multiagent auction simulation using deep reinforcement learning algorithms.

#### INF - PhD Conclusion Project
Conclusion project of postgraduate program at the Department of Informatics of PUC-Rio.

## Repository Structure
```bash
multiagent_auction/
├── models/                      # Saved model checkpoints
│   ├── actor
│   ├── critic
├── results/                     # Experiment logs, metrics, and plots
│   ├── all_pay
│   ├── common_value
│   ├── first_price
│   ├── gifs
│   ├── second_price
│   ├── tariff_discount
├── src/multiagent_auction/
│   ├── agent.py                 # Agent class with actor/critic networks
│   ├── argparser.py             # Handles command-line argument parsing
│   ├── buffer.py                # Replay buffer for experience storage
│   ├── clear.py                 # Utility script for clearing/resetting stored data
│   ├── env.py                   # Auction environment and reward mechanics
│   ├── evaluation.py            # Implements evaluation routines
│   ├── experiment.py            # Manages experiment orchestration
│   ├── maddpg.py                # MADDPG algorithm implementation
│   ├── networks.py              # Defines actor and critic networks
│   ├── run.py                   # Main entry point for training or evaluation
│   ├── train.py                 # Implements the training loop
│   ├── utils.py                 # Utility functions
├── .gitignore
├── LICENSE
└── README.md
├── params.json                  # Experiment configuration file
├── pyproject.toml
├── requirements.txt
```

## Installation and Execution

#### Installing in Anaconda environment

We can use Anaconda to set an environment and download the library.

```bash
conda create -n <environment_name> python=3.7.6
conda activate <environment_name>
pip install multiagent_auction
```
Locate the project's root directory and use pip to install the requirements (`requirements.txt`).
```bash
pip install -r requirements.txt
```
To execute the program, just type the following line on the root directory 
```bash
python src/multiagent_auction/run.py
```

## Configuration
The file ``params.json`` controls the main settings.
- auction: The auction mechanism to be used.
- target_auction: The target auction for transfer learning.
- batch: Batch size for training.
- trained: Whether to load a pre-trained model instead of starting fresh.
- gif: Whether to generate a GIF of the simulation.
- players: Number of participating players in the auction.
- noise: Standard deviation of the noise applied to bidding strategies.
- all_pay_exponent: Exponent factor used in all-pay auction variants.
- aversion_coef: Risk aversion coefficient for player strategies.
- save: Whether to save plots or trained models.
- transfer_learning: Whether to enable transfer learning between auctions.
- extra_players: Number of additional players beyond the default setup.
- show_gui: Whether to display a graphical user interface during simulation.

## Acknowledgement

This algorithm development is based on the OpenAI's DDPG algorithm. The code is inherit by [DDPG](https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg.py).
