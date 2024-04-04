import spacy
import re
import string
import nltk
from nltk.corpus import stopwords
import html 
from typing import Union, List, Tuple
import pandas as pd
from tqdm.notebook import tqdm
from emoji import demojize
from concurrent.futures import ProcessPoolExecutor, as_completed

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


# Preprocessing functions

def clean_text_part(key_text_tuple, clean_emojis=False):
    """
    Process a single document along with its unique key.
    """
    key, text = key_text_tuple

    # Handle emojis
    if clean_emojis:
        text = re.sub(r':[^:]+:', '', demojize(text))  # Remove emojis
    else:
        text = demojize(text)  # Convert emojis to text

    # Tokenization and preprocessing
    tokens = [token.text.lower() for token in nlp(text) if token.text.isalpha()]

    # Removing stopwords and short tokens
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]

    # Return a tuple of the key and the joined tokens as the processed text
    return (key, ' '.join(tokens))


def emotion_clean_text_parallel(texts_with_keys: List[Tuple[int, str]], clean_emojis=False) -> List[Tuple[int, str]]:
    """
    Process a list of documents in parallel, preserving the order by keys.
    """
    # Initialize a list to hold the processed texts with keys
    cleaned_texts_with_keys = []
    
    # Determine the number of partitions or workers
    num_partitions = 4  # Example: number of CPU cores

    # Create a pool of processes
    with ProcessPoolExecutor(max_workers=num_partitions) as executor:
        
        # Map the `clean_text_part` function to each tuple of key and text
        futures = [executor.submit(clean_text_part, key_text, clean_emojis) for key_text in texts_with_keys]
        
        # Progress bar setup
        for future in tqdm(as_completed(futures), total=len(futures), desc="Cleaning Texts"):
            # Append result maintaining the order
            cleaned_texts_with_keys.append(future.result())
    
    # Return the list of tuples with keys and processed texts
    return cleaned_texts_with_keys



def lda_clean_text(texts:Union[str, List[str], pd.Series], clean_emojis:bool=False) -> Union[str, List[str]]:

    # Create a list to store the cleaned texts
    cleaned_texts = []

    # Go through every text in the iput list of texts
    for doc in tqdm(nlp.pipe(texts, batch_size=50), 
                             total=len(texts), desc="Cleaning Texts"): 
        
        # print("Original text: ", doc)
        
        # Demojize the token.lemma for each token if it exists, else the token.text 
        tokens = [demojize(token.lemma_ if token.lemma_ != '-PRON-' else token.text).lower() for token in doc]

        # Convert emojis of form :emojiname: to words in format emojiEmojiName
        tokens = [re.sub(r':', '_', token) if token.startswith(':') and token.endswith(':') else token for token in tokens]

        # Remove emojis if prompted 
        if clean_emojis:
            tokens = [re.sub(r'_.*_', '', token) for token in tokens]

        # Remove non-alphabetic characters except for _ 
        tokens = [re.sub(r'[^a-z_]', '', token) for token in tokens]

        # Remove stopwordsm empty tokens and tokens with length less than 2
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]

        # # Join tokens that start with "no" or "not" to the next token, but preserve the original token too
        # tokens = [tokens[i] + '_' + tokens[i+1] if tokens[i] in negation_words else tokens[i] for i in range(len(tokens)-1)]
        
        # Append token to the cleaned_texts list
        cleaned_texts.append(tokens)

    # Form bigrams and trigrams models
    bigram = gensim.models.Phrases(cleaned_texts, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)  # Create bigrams with a high threshold for fewer phrases
    trigram = gensim.models.Phrases(bigram[cleaned_texts], threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)  # Create trigrams based on the bigrams
    bigram_mod = gensim.models.phrases.Phraser(bigram)  # Convert bigram model into a more efficient Phraser object
    trigram_mod = gensim.models.phrases.Phraser(trigram)  # Convert trigram model into a Phraser object for efficiency

    # Form bigrams and trigrams
    cleaned_texts = [bigram_mod[doc] for doc in tqdm(cleaned_texts, desc="creating bigrams...")]
    cleaned_texts = [trigram_mod[bigram_mod[doc]] for doc in tqdm(cleaned_texts, desc="creating trigrams...")]

    return cleaned_texts