### --- Load modules --- ###

# General modules
import timeit
import numpy as np
import matplotlib.pyplot as plt
from playsound import playsound
from datetime import timedelta

# Local modules
from utils import *
from train import *
from agent import Agent
from env import *
from evaluation import *
from argparser import parse_args

if __name__ == "__main__":
    ### --- Parsing arguments --- ###

    auction, BS, trained, n_episodes, create_gif, N, ponderated_avg, aversion_coef, save_plot, alert, z = parse_args() # get parameters

    ### --- Creating environment --- ###

    if auction == 'first_price':
        multiagent_env = MAFirstPriceAuctionEnv(N)
    elif auction == 'second_price':
        multiagent_env = MASecondPriceAuctionEnv(N)
    elif auction == 'common_value':
        vl, vh, eps = 0, 3, 0.4
        multiagent_env = MACommonPriceAuctionEnv(N, vl=vl, vh=vh, eps=eps)
    elif auction == 'tariff_discount':
        max_revenue = 5
        multiagent_env = MATariffDiscountEnv(N, max_revenue=max_revenue)
    elif auction == 'all_pay':
        multiagent_env = MAAllPayAuctionEnv(N)


    ### --- Creating agents --- ###

    agents = [Agent(alpha=0.000025, beta=0.00025, input_dims=[1], tau=0.001, 
                    env=multiagent_env, batch_size=BS, layer1_size=400, layer2_size=400, 
                    n_actions=1, total_eps=n_episodes) for i in range(N)]


    ### --- Training step --- ###

    save_interval = 10

    if not trained: # Train models if trained==False
        print('Training models...')
        if auction == 'common_value':
            score_history = MAtrainLoopCommonValue(agents, multiagent_env, n_episodes, auction, vl=vl, vh=vh, eps=eps, save_interval=save_interval)
        elif auction == 'tariff_discount':
            score_history = MAtrainLoop(agents, multiagent_env, n_episodes, auction, r=aversion_coef, max_revenue=max_revenue, gif=create_gif, save_interval=save_interval)
        else:
            score_history = MAtrainLoop(agents, multiagent_env, n_episodes, auction, r=aversion_coef, gif=create_gif, save_interval=save_interval)
        playsound('beep.mp3') if alert else None # beep when training is done    

    # Else, load models
    else:
        print('Loading models...') 
        for k in range(N):
            string = auction + '_ag' + str(k) + '_r' + str(aversion_coef) + '_' + str(n_episodes) + 'ep'
            agents[k].load_models(string)
            

        ### --- Tranfer learning step --- ###
        
        # multiagent_env = MASecondPriceAuctionEnv(N)
        # score_history = MAtrainLoop(agents, multiagent_env, n_episodes, auction='second_price', r=1)
        # playsound('beep.mp3') if alert else None # beep when training is done    


        ### --- Evaluation step --- ###
        N = 100
        evaluate(agents, N)