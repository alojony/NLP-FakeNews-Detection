#Imports

from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np
nltk.download('vader_lexicon')

# Initialize VADER
sid = SentimentIntensityAnalyzer()

def get_sentiment(df):
    ### TextBlob ###

    # Copy the dataframe to avoid modifying the original one
    df_tb = df[['key', 'author', 'tweet', 'target']].copy()
    
    # Using TextBlob to obtain polarity scores 
    def analyze_sentiment(text):
        polarity = TextBlob(str(text)).sentiment.polarity
        if polarity > 0:
            sentiment = 'positive'
        elif polarity < 0:
            sentiment = 'negative'
        else: 
            sentiment = 'neutral'
        return polarity, sentiment

    # Apply the function to the tweet column and create 2 new columns for the results 
    df_tb[['polarity_score_TB', 'tone']] = df_tb['tweet'].apply(lambda x: pd.Series(analyze_sentiment(x)))
    
    ### VADER ###
   
    # Initialize VADER
    sid = SentimentIntensityAnalyzer()
    
    # Copy the dataframe to avoid modifying the original one
    df_vader = df[['key', 'author', 'tweet', 'target']].copy()
    
    # Using VADER to obtain polarity scores 
    # Apply SID to each tweet and expand the results into separate columns
    df_vader[['neg', 'neu', 'pos', 'compound']] = df_vader['tweet'].apply(lambda x: pd.Series(sid.polarity_scores(str(x))))
    
    ### MERGE ###
    
   # Merge the TextBlob and VADER dataframes on the 'key', 'author', 'tweet', and 'target' columns
    merged_df = pd.merge(df_tb, df_vader, on=['key', 'author', 'tweet', 'target'], how='inner', suffixes=('', '_vader'))

    # Calculate the difference between 'polarity_score_TB' and 'compound'
    merged_df['score_difference'] = merged_df['polarity_score_TB'] - merged_df['compound']

    # Adding column for absolute value of compound entries 
    merged_df['absolute_polarity_V'] = merged_df['compound'].abs()

    # Ensure column names are consistent and not modified by the merge operation
    # Select the relevant columns to display
    result_df = merged_df[['key', 'author', 'tweet', 'tone', 'neg', 'neu', 'pos', 'polarity_score_TB', 'compound', 'absolute_polarity_V', 'score_difference', 'target']].copy()

    # Renaming columns if needed to match the original DataFrame structure or your requirements
    result_df.columns = ['key', 'Author', 'Tweet', 'Tone', 'Negative', 'Neutral', 'Positive', 'Polarity Score TB', 'Polarity Score Vader', 'Abs Polarity Vader', 'Score Difference', 'Target']

    return result_df
