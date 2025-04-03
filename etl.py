import pandas as pd
import numpy as np


def extract_data(file_list):
    dataframes = {}  # Initialize an empty dictionary to store DataFrames
    for filename in file_list:  # Iterate through the list of file names
        try:
            df = pd.read_parquet(filename)  # Read the Parquet file into a DataFrame
            dataframes[filename.split(".")[0]] = df  # Store the DataFrame in the dictionary, using the filename (without .parquet) as the key
        except FileNotFoundError:
            print(f"File '{filename}' not found.")  # Handle the case where the file is not found
        except Exception as e:
            print(f"An error occurred while reading {filename}: {e}")  # Handle other potential errors
    return dataframes  # Return the dictionary of DataFrames

def transform_transaction_data(transaction_df):

    # 1. Handle Missing Values
    # Replace NaN values in numerical columns with 0.
    numeric_cols = transaction_df.select_dtypes(include=np.number).columns
    transaction_df[numeric_cols] = transaction_df[numeric_cols].fillna(0)

    # Replace NaN values in categorical columns with 'unknown'.
    categorical_cols = transaction_df.select_dtypes(include='object').columns
    transaction_df[categorical_cols] = transaction_df[categorical_cols].fillna('unknown')

    # 2. Convert Data Types
    # Convert timestamp to datetime.
    transaction_df['timestamp'] = pd.to_datetime(transaction_df['timestamp'])

    # 3. Calculate Derived Features
    # Calculate transaction_fee_usd.
    transaction_df['transaction_fee_usd'] = transaction_df['transaction_fee'] * transaction_df['amount_usd'] / transaction_df['amount']

    # 4. Feature Engineering
    # Extract hour and day of week from timestamp.
    transaction_df['hour'] = transaction_df['timestamp'].dt.hour
    transaction_df['day_of_week'] = transaction_df['timestamp'].dt.dayofweek

    # 5. Handle Suspicious Transactions
    # Flag transactions with unusually high gas prices or amounts.
    transaction_df['is_high_gas'] = transaction_df['gas_price'] > transaction_df['gas_price'].quantile(0.95)
    transaction_df['is_high_amount'] = transaction_df['amount'] > transaction_df['amount'].quantile(0.95)

    # 6. Clean order id column
    transaction_df['order_id'] = transaction_df['order_id'].astype(str)

    # 7. Clean leverage column
    transaction_df['leverage'] = transaction_df['leverage'].fillna(1) # fill nan with 1.

    return transaction_df

def transform_user_data(user_df):
   # 1. Handle Missing Values
    # Replace NaN values in numerical columns with 0.
    numeric_cols = user_df.select_dtypes(include=np.number).columns
    user_df[numeric_cols] = user_df[numeric_cols].fillna(0)

    # Replace NaN values in categorical columns with 'unknown'.
    categorical_cols = user_df.select_dtypes(include='object').columns
    user_df[categorical_cols] = user_df[categorical_cols].fillna('unknown')

    # 2. Convert Data Types
    # Convert registration_date and last_login_time to datetime.
    user_df['registration_date'] = pd.to_datetime(user_df['registration_date'])
    user_df['last_login_time'] = pd.to_datetime(user_df['last_login_time'])

    # 3. Feature Engineering
    # Calculate days since registration.
    user_df['days_since_registration'] = (pd.to_datetime('now') - user_df['registration_date']).dt.days

    # Calculate days since last login.
    user_df['days_since_last_login'] = (pd.to_datetime('now') - user_df['last_login_time']).dt.days

    # 4. Handle Account Balance
    # Ensure account balance is non-negative.
    user_df['account_balance'] = user_df['account_balance'].apply(lambda x: max(0, x))

    # 5. Handle is_bot column.
    user_df['is_bot'] = user_df['is_bot'].astype(str)

    # 6. Handle referral id column.
    user_df['referral_id'] = user_df['referral_id'].astype(str)

    # 7. Handle api key usage column.
    user_df['api_key_usage'] = user_df['api_key_usage'].astype(str)

    return user_df

def transform_market_data(market_df):

    numeric_cols = market_df.select_dtypes(include=np.number).columns
    market_df[numeric_cols] = market_df[numeric_cols].fillna(0)

    # Replace NaN values in categorical columns with 'unknown'.
    categorical_cols = market_df.select_dtypes(include='object').columns
    market_df[categorical_cols] = market_df[categorical_cols].fillna('unknown')

    # 2. Convert Data Types
    # Convert market_timestamp to datetime.
    market_df['market_timestamp'] = pd.to_datetime(market_df['market_timestamp'])

    # 3. Feature Engineering
    # Calculate price change percentage.
    market_df['price_change_percentage_24h'] = (market_df['close_price'] - (market_df['close_price'] - market_df['price_change_24h'])) / (market_df['close_price'] - market_df['price_change_24h']) * 100
    market_df['price_change_percentage_7d'] = (market_df['close_price'] - (market_df['close_price'] - market_df['price_change_7d'])) / (market_df['close_price'] - market_df['price_change_7d']) * 100

    # 4. Handle extreme outliers.
    market_df = market_df[market_df['volume'] > 0]
    market_df = market_df[market_df['market_cap'] > 0]

    return market_df

def transform_network_data(network_df):
    """Transforms network data."""
    # Add your network data transformation logic here
    # Example: Convert network_timestamp to datetime
    network_df['network_timestamp'] = pd.to_datetime(network_df['network_timestamp'])
    return network_df

def transform_data(dataframes):
    """Transforms all DataFrames."""
    # 1. Handle Missing Values
    # Replace NaN values in numerical columns with 0.
    numeric_cols = network_df.select_dtypes(include=np.number).columns
    network_df[numeric_cols] = network_df[numeric_cols].fillna(0)

    # Replace NaN values in categorical columns with 'unknown'.
    categorical_cols = network_df.select_dtypes(include='object').columns
    network_df[categorical_cols] = network_df[categorical_cols].fillna('unknown')

    # 2. Convert Data Types
    # Convert network_timestamp to datetime.
    network_df['network_timestamp'] = pd.to_datetime(network_df['network_timestamp'])

    # 3. Feature Engineering
    # Calculate transactions per second.
    network_df['transactions_per_second'] = network_df['total_transactions'] / network_df['average_block_time']

    # 4. Handle outliers.
    network_df = network_df[network_df['average_block_time'] > 0]
    network_df = network_df[network_df['hashrate'] > 0]

    return network_df

def load_data(dataframes, output_filenames):
    """
    Loads DataFrames into Parquet files.

    Args:
        dataframes (dict): A dictionary of DataFrames to be loaded.
        output_filenames (dict): A dictionary of output filenames corresponding to the DataFrames.
    """
    for key, df in dataframes.items():
        try:
            df.to_parquet(output_filenames[key])
            print(f"DataFrame '{key}' loaded to {output_filenames[key]}")
        except Exception as e:
            print(f"An error occurred while loading {key}: {e}")


if __name__ == "__main__":
    file_list = ["transactions.parquet", "users.parquet", "market.parquet", "network.parquet"]
    dataframes = extract_data(file_list)
    transformed_dataframes = transform_data(dataframes)

    # Display DataFrame heads after transformation
    print("\n--- Transformed Transactions DataFrame Head ---")
    if "transactions" in transformed_dataframes:
        print(transformed_dataframes["transactions"].head())
    else:
        print("Transactions DataFra me not found.")

    print("\n--- Transformed Users DataFrame Head ---")
    if "users" in transformed_dataframes:
        print(transformed_dataframes["users"].head())
    else:
        print("Users DataFrame not found.")

    print("\n--- Transformed Market DataFrame Head ---")
    if "market" in transformed_dataframes:
        print(transformed_dataframes["market"].head())
    else:
        print("Market DataFrame not found.")

    print("\n--- Transformed Network DataFrame Head ---")
    if "network" in transformed_dataframes:
        print(transformed_dataframes["network"].head())
    else:
        print("Network DataFrame not found.")

    output_filenames = {
        "transactions": "transformed_transactions.parquet",
        "users": "transformed_users.parquet",
        "market": "transformed_market.parquet",
        "network": "transformed_network.parquet",
    }
    load_data(transformed_dataframes, output_filenames)