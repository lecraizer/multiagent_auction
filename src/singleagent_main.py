### --- Load modules --- ###

# General modules
import timeit
import numpy as np
import matplotlib.pyplot as plt
from playsound import playsound
from datetime import timedelta

# Local modules
from singleagent_train import *
from agent import Agent
from singleagent_env import *
from argparser import parse_args

if __name__ == "__main__":
    ### --- Parsing arguments --- ###

    auction, BS, trained, n_episodes, create_gif, N, ponderated_avg, aversion_coef, save_plot, alert, z = parse_args() # get parameters

    ### --- Creating environment --- ###

    if auction == 'first_price':
        env = FirstPriceAuctionEnv(N)
    elif auction == 'second_price':
        env = SecondPriceAuctionEnv(N)
    elif auction == 'all_pay':
        env = AllPayAuctionEnv(N)


    ### --- Creating single agent --- ###

    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[1], tau=0.001, 
                    env=env, batch_size=BS, layer1_size=400, layer2_size=400, 
                    n_actions=1, total_eps=n_episodes)


    ### --- Training step --- ###

    save_interval = 10
    score_history = trainLoop(agent, env, n_episodes, N=N, 
                              auction_type=auction, save_interval=1000)
    playsound('beep.mp3') if alert else None # beep when training is done    