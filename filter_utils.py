import pandas as pd

def filter_dataframe(df, filter_dict):
    """
    Filter a DataFrame based on a dictionary of feature questions and answers.
    """
    filtered_df = df.copy()

    for feature, answer in filter_dict.items():
        if feature in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[feature] == answer]

    return filtered_df