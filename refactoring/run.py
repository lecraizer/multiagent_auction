from parameters import get_parameters
from experiment import AuctionSimulationRunner

def main() -> None:
    """
    Entry point for running the auction simulation.
    Retrieves parameters, initializes the simulation runner and executes the simulation.
    """
    args = get_parameters()
    runner = AuctionSimulationRunner(*args)
    runner.execute()

if __name__ == '__main__':
    main()