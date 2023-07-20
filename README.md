# About
Multiagent auction simulation using deep reinforcement learning algorithms

# Deep RL in Auction Theory 

#### INF - PhD Conclusion Project

Conclusion project of postgraduate program at the Department of Informatics of PUC-Rio.


### Installation and Execution

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
python src/main.py -a <type of auction> -b <batch size> -d <load trained models> -e <number of episodes> -n <number of players> -p <ponderated average size> -r <aversion coefficient> -s <save test results in a plot> -t <use alert .mp3 file> -z <number of executions> 
```
where the arguments may be passed after the __main.py__ call, as described above, otherwise the default parameters will be selected

#### Examples of results

First price auction, with N=2 players, r=1 (aversion coefficient), trained in 30k episodes.

![Alt Text](results/first_price/N=2/test30k_ag1.png)

Second price auction, with N=2 players, r=1 (aversion coefficient), trained in 10k episodes.

![Alt Text](results/second_price/N=2/test10k_ag1.png)
