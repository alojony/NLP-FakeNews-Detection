import spacy
import re
import string
import nltk
from nltk.corpus import stopwords
import html 

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")  

symbol_to_word = {
        '%': 'percent',
        '$': 'dollar',
        '&': 'and',
        '?': 'question',
        '#': 'hashtag',
        '@': 'mention',
        '!': 'exclamation'}

def split_symbol(word, symbol_to_word):
    for symbol, word_replacement in symbol_to_word.items():
        word = word.replace(symbol, f" {word_replacement} ")
    return word.strip()  # Remove any leading or trailing spaces that might have been added

contractions_dict = {
    "I'm": "I am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "I've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "I'd": "I would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "I'll": "I will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not"
}


def spacy_tokenizer(text):
    nlp = spacy.load("en_core_web_sm")
    return [tok.text for tok in nlp.tokenizer(str(text))]

def text_edit(dataset,
              grp_num=False,
              rm_newline=False,
              rm_punctuation=False,
              rm_stop_words=False, 
              lowercase=False, 
              lemmatize=False, 
              expand=False, 
              html_=False,
              symb_to_text=False):
    
    stop_words = set(stopwords.words('english')) #define set of english stop words

    for attrs in dataset.values():
        #convert html
        if html_:
            attrs['tweet'] = html.unescape(attrs['tweet'])

    for attrs in dataset.values():
        if lowercase:
            attrs['tweet'] = attrs['tweet'].lower()

    for attrs in dataset.values():
        #perform expansion
        if expand:
            words = attrs['tweet'].split()
            expanded_words = []
            for word in words:
                expanded_word = contractions_dict.get(word, word)
                expanded_words.append(expanded_word)
            attrs['tweet'] = ' '.join(expanded_words)

    for attrs in dataset.values():
        #stop words removal
        if rm_stop_words:
            tweet_words = attrs['tweet'].split()
            filtered_tweet = ' '.join([word for word in tweet_words if word not in stop_words]) 
            attrs['tweet'] = filtered_tweet

    for attrs in dataset.values():
        #all numbers to 'num'
        if grp_num:
            attrs['tweet'] = re.sub(r'\d+', 'num', attrs['tweet'])  

    for attrs in dataset.values():
        # Perform symbol_to_text conversion
        if symb_to_text:
            words = attrs['tweet'].split()
            attrs['tweet'] = ' '.join([split_symbol(word, symbol_to_word) for word in words])

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