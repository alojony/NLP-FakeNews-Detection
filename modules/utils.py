import pandas as pd

mapping = {'politicsNews': 0, 'worldnews': 1, 'News': 2, 'politics': 3, 'left-news': 4, 
               'Government News': 5, 'US_News': 6, 'Middle-east': 7}

def build_dataset(true_path, false_path, num_class_samples=-1):
    true_df = pd.read_csv(true_path)
    true_df['is_real'] = 1
    false_df = pd.read_csv(false_path)
    false_df['is_real'] = 0
    false_df['text'] = false_df['text'].str.split('-').apply(lambda x: ' '.join(x[1:]))

    if num_class_samples == -1:
        true_sampled_df = true_df
        false_sampled_df = false_df
    else:
        true_sampled_df = true_df.sample(n=min(len(true_df), num_class_samples), replace=False, random_state=42)
        false_sampled_df = false_df.sample(n=min(len(false_df), num_class_samples), replace=False, random_state=42)

    df = pd.concat([true_sampled_df, false_sampled_df], ignore_index=True)

    df['subject'] = df['subject'].map(mapping)

    return df.T.to_dict()
