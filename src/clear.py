import os


def clear_data():
    '''
    Clear data from previous runs
    '''
    auction_types = ['first_price', 'second_price', 'common_value', 'tariff_discount']
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

if __name__ == "__main__":
    main()