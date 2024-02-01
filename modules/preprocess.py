import spacy


def spacy_tokenizer(text, lowercase=False):
    nlp = spacy.load("en_core_web_sm")
    if lowercase:
        return [tok.text.lower() for tok in nlp.tokenizer(str(text))]
    else:
        return [tok.text for tok in nlp.tokenizer(str(text))]
    