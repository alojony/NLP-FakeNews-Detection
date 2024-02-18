import pandas as pd

def build_dataset(path, num_class_samples=-1, rnd_state=42):
    df = pd.read_excel(path, engine='openpyxl')
    if num_class_samples != -1:
        df = df.sample(n=min(len(df), num_class_samples), replace=False, random_state=rnd_state)
    return df.T.to_dict()
