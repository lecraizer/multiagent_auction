# Script to clear up saved models and results after test runs

import os
import argparse

def clear_data():
    '''
    Clears data from previous runs.

    Deletes data in the following directories:
    - results/{auction_type}/
    - models/actor/
    - models/critic/

    Prints a success message after the data is cleared.
    '''
    for auc_type in ['first_price', 'second_price', 'common_value', 'tariff_discount', 'all_pay', 'core_selecting']:
        os.system('rm -rf results/' + auc_type + '/*')
        os.system('rm -rf models/actor/' + '*')
        os.system('rm -rf models/critic/' + '*')
    print("\nData cleared successfully.")

def main():
    '''
    Main function to clear data.

    Asks the user if they are sure they want to clear the data. 
    If the user confirms (by entering 'yes' or pressing Enter), it calls clear_data function. 
    If the user enters 'no', a message is printed stating that the data was not cleared.
    If the input is invalid, an error message is shown.

    Command-line arguments:
    - -g, --gif (optional): Clears the 'gifs/' folder.
    '''
    user_input = input("Are you sure you want to clear data? (Yes/No): ").lower()

    # If the user enters "yes" or simply presses Enter (default value)
    if user_input == "yes" or not user_input:
        clear_data()
    elif user_input == "no":
        print("Data was not cleared.")
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")

    parser = argparse.ArgumentParser(description='Parse terminal input information')
    parser.add_argument('-g', '--gif', type=str, help='Clean gifs folder')
    args = parser.parse_args()

    if args.gif: # g
        os.system('rm -rf gifs/' + '*')
        print("\nGifs cleared successfully.")

if __name__ == "__main__":
    main()