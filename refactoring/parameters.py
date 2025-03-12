import os
from dotenv import load_dotenv

def get_parameters():
    parameters = {'AUCTION': 'first_price',
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

    for param in list(parameters.keys()):
        if os.getenv(param) != None:
            if isinstance(parameters[param], int):
                parameters[param] = int(os.getenv(param))
            elif isinstance(parameters[param], float):
                parameters[param] = float(os.getenv(param))
            elif isinstance(parameters[param], bool):
                parameters[param] = bool(os.getenv(param))
            else:
                parameters[param] = os.getenv(param)

    print('Auction type: ', parameters['AUCTION'])
    print('Batch size: ', parameters['BATCH'])
    print('Load models: ', parameters['TRAINED'])
    print('Number of episodes: ', parameters['EPISODES'])
    print('Create gif: ', parameters['GIF'])
    print('Number of players: ', parameters['PLAYERS'])
    print('Noise standard deviation: ', parameters['NOISE'])
    print('Ponderated average size: ', parameters['PONDERATED'])
    print('Aversion coefficient: ', parameters['AVERSION_COEF'])
    print('Save plot: ', parameters['SAVE'])
    print('Alert: ', parameters['ALERT'])
    print('Transfer learning: ', parameters['TRANSFER_LEARNING'])
    print('Extra players: ', parameters['EXTRA_PLAYERS'])
    print('Number of executions: ', parameters['EXECUTIONS'])
    print('\n')

    return list(parameters.values())