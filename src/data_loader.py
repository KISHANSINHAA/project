import pandas as pd
import os


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw retail sales dataset and perform basic cleaning.

    Parameters:
        file_path (str): Path to raw CSV file

    Returns:
        pd.DataFrame: Cleaned raw dataframe
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    return df


def create_daily_aggregated_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data into daily-level data.

    Parameters:
        df (pd.DataFrame): Transaction-level dataframe

    Returns:
        pd.DataFrame: Daily aggregated dataframe
    """

    daily_df = (
        df.groupby("date")
        .agg(
            daily_revenue=("total_amount", "sum"),
            num_transactions=("transaction_id", "count"),
            total_quantity=("quantity", "sum"),
            unique_customers=("customer_id", "nunique")
        )
        .reset_index()
        .sort_values("date")
    )

    return daily_df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed dataframe to CSV.

    Parameters:
        df (pd.DataFrame): Dataframe to save
        output_path (str): Destination path
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}")


# Optional: Run as script
if __name__ == "__main__":
    RAW_DATA_PATH = "data/raw/retail_sales_dataset.csv"
    OUTPUT_PATH = "data/processed/daily_data.csv"

    raw_df = load_raw_data(RAW_DATA_PATH)
    daily_df = create_daily_aggregated_data(raw_df)
    save_processed_data(daily_df, OUTPUT_PATH)

    print("✅ Data loading and aggregation completed successfully")
