import numpy as np
from multiagent_auction.env import *
from multiagent_auction.train import *
from multiagent_auction.evaluation import *
from multiagent_auction.maddpg import MADDPG

class AuctionSimulationRunner:
    def __init__(self, 
                 auction: str,  
                 target_auction: str,
                 batch_size: int, 
                 trained: bool, 
                 n_episodes: int, 
                 create_gif: bool, 
                 n_players: int, 
                 t: float,
                 noise_std: float,
                 ponderated_avg: bool, 
                 aversion_coef: float,
                 save_plot: bool,
                 tl: bool, 
                 extra_players: int, 
                 gui: bool):
        """
        Initialize the simulation runner with configuration parameters.

        Args:
            auction (str): Auction type to simulate.
            batch_size (int): Batch size for training.
            trained (bool): Whether to evaluate pre-trained models or train new ones.
            n_episodes (int): Number of training episodes.
            create_gif (bool): Whether to save training GIFs.
            n_players (int): Initial number of agents.
            noise_std (float): Standard deviation for action noise.
            ponderated_avg (bool): Use ponderated averages in evaluation.
            aversion_coef (float): Risk aversion coefficient.
            save_plot (bool): Whether to save evaluation plots.
            tl (bool): Enable transfer learning.
            extra_players (int): Number of agents to add incrementally via transfer learning.
        """
        self.auction = auction
        self.target_auction = target_auction
        self.batch_size = batch_size
        self.trained = trained
        self.n_episodes = n_episodes
        self.create_gif = create_gif
        self.n_players = n_players
        self.t = t
        self.noise_std = noise_std
        self.ponderated_avg = ponderated_avg
        self.aversion_coef = aversion_coef
        self.save_plot = save_plot
        self.tl = tl
        self.extra_players = extra_players
        self.gui = gui

    def create_env(self, N: int):
        """
        Create and return a multi-agent auction environment based on the auction type.

        Args:
            N (int): Number of agents in the environment.

        Returns:
            Auction environment instance.

        Raises:
            ValueError: If the auction type is not recognized.
        """
        if self.auction == 'first_price':
            return MAFirstPriceAuctionEnv(N)
        elif self.auction == 'second_price':
            return MASecondPriceAuctionEnv(N)
        elif self.auction == 'all_pay':
            return MAAllPayAuctionEnv(N)
        elif self.auction == 'partial_all_pay':
            return MAPartialAllPayAuctionEnv(N)
        elif self.auction == 'custom':
            return MACustomAuctionEnv(N)
        else:
            raise ValueError(f"Auction type '{self.auction}' not recognized.")

    def load_agents(self, maddpg: MADDPG, N: int) -> None:
        """
        Load pre-trained models for each agent.

        Args:
            maddpg (MADDPG): Trainer instance containing agent models.
            N (int): Number of agents.
        """
        for k in range(N):
            name = f"{self.auction}_N_{N}_ag{k}_r{self.aversion_coef}_{self.n_episodes}ep"
            maddpg.agents[k].load_models(name)

    def initialize_new_agent_from_random(self, maddpg: MADDPG, new_agent_idx: int) -> None:
        """
        Initialize a new agent's networks by copying weights from a randomly selected existing agent.

        Args:
            maddpg (MADDPG): Trainer instance containing agent models.
            new_agent_idx (int): Index of the new agent to initialize.
        """
        rd_agt_idx = np.random.randint(0, new_agent_idx)
        maddpg.agents[new_agent_idx].actor.load_state_dict(maddpg.agents[rd_agt_idx].actor.state_dict())
        maddpg.agents[new_agent_idx].critic.load_state_dict(maddpg.agents[rd_agt_idx].critic.state_dict())
        maddpg.agents[new_agent_idx].target_actor.load_state_dict(maddpg.agents[rd_agt_idx] \
                                                                  .target_actor.state_dict())
        maddpg.agents[new_agent_idx].target_critic.load_state_dict(maddpg.agents[rd_agt_idx] \
                                                                   .target_critic.state_dict())

    def execute(self) -> None:
        """
        Run the auction simulation: training agents or evaluating pre-trained ones.
        """
        env = self.create_env(self.n_players)
        maddpg = MADDPG(alpha=0.000025, beta=0.00025, input_dims=1, tau=0.001, gamma=0.99, BS=self.batch_size,
                         fc1=100, fc2=100, n_actions=1, n_agents=self.n_players, total_eps=self.n_episodes, 
                         noise_std=0.2, tl_flag=self.tl, extra_players=self.extra_players)

        if not self.trained:
            print('Training models...')
            MAtrainLoop(maddpg, env, self.n_episodes, self.auction, t=self.t, r=self.aversion_coef, 
                        gif=self.create_gif, save_interval=50, tl_flag=self.tl, 
                        extra_players=self.extra_players, show_gui=self.gui)
            
            if self.tl and self.extra_players == 0:
                self.auction = self.target_auction
                print(f"Transferring from auction '{self.auction} \
                      ' to '{self.target_auction}' with N={self.n_players} agents...")
                env = self.create_env(self.n_players)
                MAtrainLoop(maddpg, env, self.n_episodes, self.auction, t=self.t, r=self.aversion_coef, 
                            gif=self.create_gif,save_interval=50, tl_flag=self.tl, 
                            extra_players=self.extra_players, show_gui=self.gui)

        if self.tl:
            for i in range(self.extra_players):
                new_N = self.n_players + i + 1
                prev_N = new_N - 1
                is_last = (i == self.extra_players - 1)
                tl_flag_iter = not is_last
                extra_left = self.extra_players - i - 1 if not is_last else 0

                print(f'\nTransfer learning step {i+1}/{self.extra_players}: from {prev_N} to {new_N} agents\n')

                maddpg = MADDPG(alpha=0.000025, beta=0.00025, input_dims=1, tau=0.001, gamma=0.99, 
                                BS=self.batch_size, fc1=100, fc2=100, n_actions=1,
                                n_agents=new_N, total_eps=self.n_episodes, noise_std=0.2,
                                tl_flag=tl_flag_iter, extra_players=extra_left)

                for k in range(prev_N):
                    model_name = f"{self.auction}_N_{prev_N}_ag{k}_r{self.aversion_coef}_{self.n_episodes}ep"
                    maddpg.agents[k].load_models(model_name)

                self.initialize_new_agent_from_random(maddpg, new_agent_idx=new_N - 1)

                env = self.create_env(new_N)
                MAtrainLoop(maddpg, env, self.n_episodes, self.auction,r=self.aversion_coef, 
                            gif=self.create_gif,save_interval=50, tl_flag=tl_flag_iter, 
                            extra_players=extra_left)
                
        if self.trained and not self.tl:
            print('Evaluating models...')
            self.load_agents(maddpg, self.n_players)
            evaluate_agents(env, maddpg.agents, n_bids=100, grid_precision=100, auc_type=self.auction)