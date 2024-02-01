import pandas as pd


def build_dataset(true_path, false_path, num_class_samples=-1):
    true_df = pd.read_csv(true_path)
    true_df['is_real'] = 1
    false_df = pd.read_csv(false_path)
    false_df['is_real'] = 0
    false_df['text'] = false_df['text'].str.split('-').apply(lambda x: ' '.join(x[1:]))
    df = pd.concat([true_df[:num_class_samples], false_df[:num_class_samples]], ignore_index=True)
    return df.T.to_dict()