import os
import argparse


def clear_data():
    '''
    Clear data from previous runs
    '''
    auction_types = ['first_price', 'second_price', 'common_value', 'tariff_discount', 'all_pay']
    for auc_type in auction_types:
        os.system('rm -rf results/' + auc_type + '/*')
        os.system('rm -rf models/actor/' + '*')
        os.system('rm -rf models/critic/' + '*')
    print("\nData cleared successfully.")


def main():
    user_input = input("Are you sure you want to clear data? (Y/n): ").lower()

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