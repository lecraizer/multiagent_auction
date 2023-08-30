import timeit
import numpy as np
from datetime import timedelta
from singleagent_utils import *


def trainLoop(agent, env, n_episodes, N=2, auction_type='first_price', save_interval=100):
    np.random.seed(0)
    start_time = timeit.default_timer()

    for ep in range(n_episodes):
        obs = env.reset()
        act = agent.choose_action(obs, ep)
        reward = env.step(act)
        agent.remember(obs, act, reward)
        agent.learn()

        if ep % save_interval == 0:
            print('\nEpisode:', ep)
            print('Value:  ', obs)
            print('Bid:    ', act[0])
            print('Reward: ', reward)
            manualTesting(agent, N, ep, n_episodes, auc_type=auction_type)
            
            # decrease learning rate each n episodes
            decrease_learning_rate(agent, 0.999)

    # Total training time
    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])
