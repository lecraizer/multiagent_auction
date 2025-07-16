from parameters import get_parameters
from experiment import AuctionSimulationRunner

def main():
    args = get_parameters()
    runner = AuctionSimulationRunner(*args)
    runner.execute()

if __name__ == '__main__':
    main()