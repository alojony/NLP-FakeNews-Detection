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


def text_edit(dataset, grp_num=False, rm_newline=False, rm_punctuation=False,
              rm_stop_words=False, lowercase=False, lemmatize=False,
              expand=False, html_=False, symb_to_text=False, convert_entities=False, reduce_mentions=False):

    stop_words = set(stopwords.words('english'))  # Define set of English stop words
    pattern = re.compile(f"[{re.escape(string.punctuation)}]")  # Compile regex pattern for performance

    for attrs in dataset.values():
        tweet = attrs['tweet']

        if convert_entities:
            doc = nlp(tweet)
            for ent in doc.ents:
                if ent.label_ in ['PERSON']:
                    tweet = tweet.replace(ent.text, 'person')
                elif ent.label_ in ['GPE', 'LOC']:
                    tweet = tweet.replace(ent.text, 'place')
                elif ent.label_ in ['DATE', 'TIME']:
                    tweet = tweet.replace(ent.text, 'time')

        if html_:
            tweet = html.unescape(tweet)

        if lowercase:
            tweet = tweet.lower()

        if reduce_mentions:
            tweet = re.sub(r'#\w+', 'hashtag', tweet)
            tweet = re.sub(r'@\w+', 'mention', tweet)

        if expand:
            words = tweet.split()
            expanded_words = [contractions_dict.get(word, word) for word in words]
            tweet = ' '.join(expanded_words)

        if rm_stop_words:
            words = tweet.split()
            tweet = ' '.join(word for word in words if word not in stop_words)

        if grp_num:
            tweet = re.sub(r'\d+', 'num', tweet)

        if rm_newline:
            tweet = tweet.replace('\n', '')

        if symb_to_text:
            words = tweet.split()
            tweet = ' '.join([split_symbol(word, symbol_to_word) for word in words])

        if rm_punctuation:
            tweet = pattern.sub('', tweet)
            tweet = re.sub(r' +', ' ', tweet)

        if lemmatize:
            tweet_words = tweet.split()
            tweet = ' '.join(tok.lemma_ for tok in nlp(' '.join(tweet_words)))

        # Update the tweet after all transformations
        attrs['tweet'] = tweet

    return dataset