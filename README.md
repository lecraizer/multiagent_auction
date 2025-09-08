# Deep Reinforcement Learning in Auction Theory 
Multiagent auction simulation using deep reinforcement learning algorithms

#### INF - PhD Conclusion Project
Conclusion project of postgraduate program at the Department of Informatics of PUC-Rio.

## Installation and Execution

#### Installing in Anaconda environment

We can use Anaconda to set an environment.

```bash
conda create -n <environment_name> python=3.7.6
conda activate <environment_name>
```

#### Install the dependencies of the project through the command

Then, locate the project's root directory and use pip to install the requirements (`requirements.txt`).

```bash
pip install -r requirements.txt
```

#### To execute the program, just type the following line on the root directory 

```bash
python src/run.py
```

## Acknowledgement

This algorithm development is based on the OpenAI's DDPG algorithm. The code is inherit by [DDPG](https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg.py).
