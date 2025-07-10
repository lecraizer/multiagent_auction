# Entry point for running auction experiments

from experiment import AuctionSimulationRunner
from argparser import parse_args

if __name__ == "__main__":
    args = parse_args()
    runner = AuctionSimulationRunner(*args)
    runner.execute()
