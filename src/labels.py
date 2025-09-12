# src/labels.py

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def add_labels(df, start_hour=8, end_hour=9, threshold=0.0):
    """
    Identifies trading opportunities based on price movement between start and end hours.

    This function scans a DataFrame of financial data, assuming it has a DateTimeIndex
    set to a specific timezone (e.g., 'Europe/London'). It identifies rows corresponding
    to a `start_hour` (e.g., 08:00) and looks for the corresponding `end_hour`
    (e.g., 09:00) on the same day to determine the price direction.

    Args:
        df (pd.DataFrame): Input DataFrame with a timezone-aware DateTimeIndex and columns
                           like 'Open', 'Close', and 'Ticker'.
        start_hour (int): The hour to check for a potential trade entry (e.g., 8 for 08:00).
        end_hour (int): The hour to determine the outcome of the trade (e.g., 9 for 09:00).
        threshold (float): A value to compare the price change against. The default of 0.0
                           means any positive change is 'up' (1) and any non-positive
                           change is 'down' (0).

    Returns:
        pd.DataFrame: A new DataFrame containing only the rows from the `start_hour` that
                      have a valid corresponding `end_hour` target. It includes the original
                      data plus a 'label' column indicating the price direction (1 for up,
                      0 for down). Returns an empty DataFrame if no valid labels can be generated.
    """
    df['target_dir'] = -1  # Default to -1 (no signal)

    # Get all unique days from the DataFrame index
    unique_days = df.index.normalize().unique()

    # Create target timestamps for each day
    target_starts = pd.to_datetime([f'{day.date()} {start_hour:02d}:00' for day in unique_days]).tz_localize(df.index.tz)
    target_ends = pd.to_datetime([f'{day.date()} {end_hour:02d}:00' for day in unique_days]).tz_localize(df.index.tz)

    # Filter for targets that actually exist in the index to avoid errors
    valid_starts = df.index.intersection(target_starts)
    valid_ends = df.index.intersection(target_ends)

    # Map start times to their corresponding end times
    start_to_end_map = {start: end for start, end in zip(valid_starts, target_ends) if end in valid_ends}

    if not start_to_end_map:
        logging.warning("No valid start/end pairs found. Cannot generate labels.")
        return pd.DataFrame()

    # Calculate direction for valid pairs
    for start_time, end_time in start_to_end_map.items():
        # Get start and end rows for each ticker present at both times
        start_rows = df.loc[start_time]
        end_rows = df.loc[end_time]

        # Ensure they are DataFrames for consistent processing
        if isinstance(start_rows, pd.Series):
            start_rows = start_rows.to_frame().T
        if isinstance(end_rows, pd.Series):
            end_rows = end_rows.to_frame().T

        # Find common tickers
        common_tickers = start_rows['Ticker'].isin(end_rows['Ticker'])
        valid_start_rows = start_rows[common_tickers]

        for _, row in valid_start_rows.iterrows():
            ticker = row['Ticker']
            end_close = end_rows[end_rows['Ticker'] == ticker]['Close'].iloc[0]
            price_change = end_close - row['Open']

            direction = 1 if price_change > threshold else 0
            df.loc[row.name, 'target_dir'] = direction

    # Filter for rows where a label was successfully generated
    labels = df.loc[df['target_dir'] != -1].copy()
    logging.info(f"Label samples ({start_hour:02d}:00 rows with {end_hour:02d}:00 targets): {len(labels)}")

    if labels.empty:
        logging.warning("DataFrame is empty after labeling. Check time range and data integrity.")
        return pd.DataFrame()

    # --- THIS IS THE FIX ---
    # Rename the 'target_dir' column to 'label' to match what the next script expects.
    labels.rename(columns={'target_dir': 'label'}, inplace=True)
    # -----------------------

    return labels
