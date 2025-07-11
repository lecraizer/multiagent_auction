# Entry point for running auction experiments

from experiment import AuctionSimulationRunner
from argparser import parse_args

def main():
    args = parse_args()
    runner = AuctionSimulationRunner(*args)
    runner.execute()

if __name__ == '__main__':
    main()