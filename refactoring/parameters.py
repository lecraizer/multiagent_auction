import os
from dotenv import load_dotenv

def print_parameters(default_parameters: dict) -> None:
    """
    Print all configuration parameters from the provided dictionary.
    
    Args:
        default_parameters (dict): A dictionary containing parameters 
            for the auction environment. Keys are parameter names and
            values are their respective values.
    """
    print('Auction type: ', default_parameters['AUCTION'])
    print('Batch size: ', default_parameters['BATCH'])
    print('Load models: ', default_parameters['TRAINED'])
    print('Number of episodes: ', default_parameters['EPISODES'])
    print('Create gif: ', default_parameters['GIF'])
    print('Number of players: ', default_parameters['PLAYERS'])
    print('Noise standard deviation: ', default_parameters['NOISE'])
    print('Ponderated average size: ', default_parameters['PONDERATED'])
    print('Aversion coefficient: ', default_parameters['AVERSION_COEF'])
    print('Save plot: ', default_parameters['SAVE'])
    print('Alert: ', default_parameters['ALERT'])
    print('Transfer learning: ', default_parameters['TRANSFER_LEARNING'])
    print('Extra players: ', default_parameters['EXTRA_PLAYERS'])
    print('Number of executions: ', default_parameters['EXECUTIONS'])
    print('\n')

def get_parameters() -> list:
    """
    Retrieve and configure system parameters from environment variables.
    This function creates a dictionary with default parameter values for the auction
    system, then overrides those defaults with any values found in environment
    variables. It handles type conversion for integer, float, and boolean values.
    Finally, it prints the resolved parameters and returns them as a list.
    
    Returns:
        list: A list containing all parameter values in the same order as they
            appear in the default_parameters dictionary.
    """
    default_parameters = {'AUCTION': 'first_price',
                          'BATCH': 64,
                          'TRAINED': 0,
                          'EPISODES': 2000,
                          'GIF': 0,
                          'PLAYERS': 2,
                          'NOISE': 0.2,
                          'PONDERATED': 100,
                          'AVERSION_COEF': 1,
                          'SAVE': 1,
                          'ALERT': 0,
                          'TRANSFER_LEARNING': False,
                          'EXTRA_PLAYERS': 0,
                          'EXECUTIONS': 1
                          }

    load_dotenv()

    for param in list(default_parameters.keys()):
        if os.getenv(param) != None:
            if isinstance(default_parameters[param], int):
                default_parameters[param] = int(os.getenv(param))
            elif isinstance(default_parameters[param], float):
                default_parameters[param] = float(os.getenv(param))
            elif isinstance(default_parameters[param], bool):
                default_parameters[param] = bool(os.getenv(param))
            else:
                default_parameters[param] = os.getenv(param)

    print_parameters(default_parameters)

    return list(default_parameters.values())