# General Imports
import re
from tqdm.notebook import tqdm
from emoji import demojize
from typing import Union, List

from nrclex import NRCLex

# Data Analysis and visualizations
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Import Spacy
import spacy
from spacy.lang.en import English
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

# Import NLTK
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
nltk.download('punkt', quiet=True)
nltk.download('sentiwordnet')
nltk.download('wordnet')


def analyze_tweets_emotions(processed_tweets_with_keys):
    """
    Analyze emotions in tweets along with their keys using NRCLex, returning a DataFrame with 
    the original tweet, key and the top two emotions.

    Parameters:
    - processed_tweets_with_keys: A list of tuples, each containing a key and a pre-processed tweet text.

    Returns:
    - A pandas DataFrame with columns for the key, tweet, the top two emotions, and their scores.
    """
    data_list = []

    for key, tweet in processed_tweets_with_keys:
        if pd.notnull(tweet):  # Check if the tweet is not null
            emotions = NRCLex(tweet)
            emotion_scores = emotions.raw_emotion_scores
            
            # Remove 'negative' and 'positive' from the scores if they exist
            emotion_scores.pop('negative', None)
            emotion_scores.pop('positive', None)
            
            # Sort the remaining emotions based on their frequency and pick the top two
            top_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            
            if len(top_emotions) >= 2:
                data_list.append([
                    key, tweet, 
                    top_emotions[0][0], top_emotions[0][1],  # Emotion 1 and its score
                    top_emotions[1][0], top_emotions[1][1]   # Emotion 2 and its score
                ])
            elif len(top_emotions) == 1:
                data_list.append([
                    key, tweet,
                    top_emotions[0][0], top_emotions[0][1],  
                    None, 0  # Placeholder for the second emotion
                ])
            else:
                data_list.append([
                    key, tweet,
                    None, 0,  
                    None, 0  # Placeholder if no emotions detected
                ])
    
    return pd.DataFrame(data_list, columns=['key', 'tweet', 'emotion 1', 'emotion 1 score', 'emotion 2', 'emotion 2 score'])



def plot_emotion_distribution(df, column_name):
    """
    Plot the distribution of emotions or any categorical data in a specified column of a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - column_name: String specifying the column in the DataFrame to be visualized.

    Returns:
    - A bar plot showing the distribution of values in the specified column.
    """
    plt.figure(figsize=(12, 8))  # Increase figure size
    df[column_name].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {column_name}', fontsize=16)
    plt.xlabel('Primary Emotion' if column_name.lower().startswith('emotion') else 'Category', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Adjust the layout to make room for the rotated x-axis labels
    plt.show()