import spacy
import re
import string


def spacy_tokenizer(text):
    nlp = spacy.load("en_core_web_sm")
    return [tok.text for tok in nlp.tokenizer(str(text))]



def dataset_modif(dataset):
    #convert all numbers to num
    for attrs in dataset.values():
        attrs['tweet'] = re.sub(r'\d+', 'num', attrs['tweet'])
    #convert everything beginning with @... to @
    for attrs in dataset.values():
        attrs['tweet'] = re.sub(r'@\w+', '@', attrs['tweet'])
    #remove anything with \n in it
    for attrs in dataset.values():
        attrs['tweet'] = re.sub(r'\n', '', attrs['tweet'])
    #remove all punctuation
    for attrs in dataset.values():
        pattern = f"[{re.escape(string.punctuation)}]"
        attrs['tweet'] = re.sub(pattern, '', attrs['tweet'])
    return dataset