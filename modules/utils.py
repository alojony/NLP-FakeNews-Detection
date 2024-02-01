import pandas as pd

def build_dataset(true_path, false_path):
    true_df = pd.read_csv(true_path)
    true_df['is_real'] = 1
    false_df = pd.read_csv(false_path)
    false_df['is_real'] = 0
    df = pd.concat([true_df, false_df], ignore_index=True)
    return df.T.to_dict()