import spacy
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")  


def spacy_tokenizer(text):
    nlp = spacy.load("en_core_web_sm")
    return [tok.text for tok in nlp.tokenizer(str(text))]

def text_edit(dataset,grp_num=False,grp_mention=False,rm_newline=False,rm_punctuation=False,rm_stop_words=False, lowercase=False, lemmatize=False):
    stop_words = set(stopwords.words('english')) #define set of english stop words

    for attrs in dataset.values():
        if lowercase:
            attrs['tweet'] = attrs['tweet'].lower()

    for attrs in dataset.values():
        if rm_stop_words:
            tweet_words = attrs['tweet'].split()
            filtered_tweet = ' '.join([word for word in tweet_words if word not in stop_words]) 
            attrs['tweet'] = filtered_tweet

    for attrs in dataset.values():
        #all numbers to 'num'
        if grp_num:
            attrs['tweet'] = re.sub(r'\d+', 'num', attrs['tweet'])  

    for attrs in dataset.values():
        #mentions to '@'
        if grp_mention:
            attrs['tweet'] = re.sub(r'@\w+', '@', attrs['tweet'])

    for attrs in dataset.values():
        #remove newline characters
        if rm_newline:
            attrs['tweet'] = attrs['tweet'].replace('\n', '')

    for attrs in dataset.values():
        #remove punctuation
        pattern = f"[{re.escape(string.punctuation)}]"
        if rm_punctuation:
            attrs['tweet'] = re.sub(pattern, '', attrs['tweet'])
            attrs['tweet'] = re.sub(r' +', ' ', attrs['tweet'])

    for attrs in dataset.values():
        if lemmatize:
            tweet_words = attrs['tweet'].split()
            filtered_tweet = ' '.join([tok.lemma_ for tok in nlp(' '.join(tweet_words))]) 
            attrs['tweet'] = filtered_tweet

    return dataset