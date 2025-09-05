from multiagent_auction.argparser import load_args
from multiagent_auction.experiment import AuctionSimulationRunner

def main() -> None:
    """
    Entry point for running the auction simulation.
    Retrieves parameters, initializes the simulation runner and executes the simulation.
    """
    args = load_args()
    runner = AuctionSimulationRunner(*args)
    runner.execute()

if __name__ == '__main__':
    main()