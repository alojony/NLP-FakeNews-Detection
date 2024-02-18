import spacy
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def spacy_tokenizer(text):
    nlp = spacy.load("en_core_web_sm")
    return [tok.text for tok in nlp.tokenizer(str(text))]

def text_edit(dataset,num=False,mention=False,newline=False,punctuation=False,stop_words_=False):
    stop_words = set(stopwords.words('english')) #define set of english stop words
    
    for attrs in dataset.values():
        tweet = attrs['tweet']

        #all numbers to 'num'
        if num:
            tweet = re.sub(r'\d+', 'num', tweet)  

        #mentions to '@'
        if mention:
            tweet = re.sub(r'@\w+', '@', tweet)   

        #remove newline characters
        if newline:
            tweet = tweet.replace('\n', '')       
            pattern = f"[{re.escape(string.punctuation)}]"

        #remove punctuation
        if punctuation:
            tweet = re.sub(pattern, '', tweet)

        #remove stop words
        if stop_words_:
            tweet_words = tweet.split()
            filtered_tweet = ' '.join([word for word in tweet_words if word.lower() not in stop_words]) 
            attrs['tweet'] = filtered_tweet

    return dataset