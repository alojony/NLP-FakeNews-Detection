# General imports
import pprint 
from typing import Union, List 
from tqdm.notebook import tqdm 
from collections import Counter

# Data Analysis and visualizations
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

# NLTK setup
import nltk 
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Text Processing 
import re 
import spacy
import gensim
from gensim import corpora
from emoji import demojize
from spacy.tokens import Doc
from nltk.corpus import stopwords
from gensim.models.callbacks import PerplexityMetric
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS
from gensim.models import LdaMulticore

# Dedicated NLP Visualizations 
import pyLDAvis
import pyLDAvis.gensim
from wordcloud import WordCloud

# Configurations 
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Exclude common negation words from the stop words list
# negation_words = {'no', 'not', 'nor', 'neither', 'never', "n't", 'none', 'through'}
except_words = {'through'}
stop_words = stop_words - except_words

# Load Spacy model and disable irrelevant components for acceleration
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
# nlp.max_length = 1500000  # Adjust based on your text size

# Set pprint options with indent 4
pprint = pprint.PrettyPrinter(indent=4).pprint
# Ignore warnings 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def train_and_save_lda_model(clean_texts, lda_hyperparams=None, model_dir='models/lda_model'):
    """
    Train an LDA model on a subset of clean texts and save the model.

    Parameters:
    - clean_texts: List of lists, where each sublist contains tokens from a clean text document.
    - lda_hyperparams: Dictionary of LDA hyperparameters. If None, default parameters will be used.
    - model_dir: Directory path where the LDA model will be saved.

    Returns:
    - The trained LDA model.
    """
    # Set default LDA hyperparameters if none provided
    if lda_hyperparams is None:
        lda_hyperparams = {
            'num_topics': 4,
            'update_every': 1,
            'chunksize': 50,
            'passes': 15,
            'alpha': 'symmetric',
            'iterations': 100,
        }

    # Create a subset of randomly selected clean texts
    subset_size = min(50000, len(clean_texts))
    clean_texts_subset = [clean_texts[i] for i in np.random.choice(len(clean_texts), subset_size, replace=False)]

    # Create a dictionary mapping from word IDs to words and convert to BoW
    id2word = corpora.Dictionary(clean_texts_subset)
    corpus = [id2word.doc2bow(text) for text in clean_texts_subset]

    # Initialize PerplexityMetric for logging
    perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')

    # Train LDA model
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        workers=3,  # Adjust based on your machine's capability
        callbacks=[perplexity_logger],
        **lda_hyperparams
    )

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Construct model filename from hyperparameters
    model_name = f'lda_model_{"_".join([str(v) for v in lda_hyperparams.values()])}.model'
    model_path = os.path.join(model_dir, model_name)

    # Save the model
    lda_model.save(model_path)

    # Print the keywords in the topics
    pprint(lda_model.print_topics())

    return lda_model

## Analysis

def find_representative_texts(df_dominant_topic):
    """
    Identify the most representative text for each topic based on the highest percentage contribution.

    Parameters:
    - df_dominant_topic: DataFrame with columns 'Dominant_Topic' and 'Perc_Contribution',
                         among others, containing information on each document's dominant topic
                         and its contribution percentage.

    Returns:
    - A DataFrame with the most representative text for each topic, sorted by 'Perc_Contribution'.
    """
    # Increase the maximum width of columns in pandas DataFrame displays
    pd.options.display.max_colwidth = 100

    # Initialize an empty DataFrame to store sorted topic information
    sent_topics_sorteddf_mallet = pd.DataFrame()

    # Group the DataFrame by the 'Dominant_Topic' column
    sent_topics_outdf_grpd = df_dominant_topic.groupby('Dominant_Topic')

    # Iterate over each group representing a single dominant topic
    for _, grp in sent_topics_outdf_grpd:
        # Sort the grouped DataFrame by 'Perc_Contribution' in descending order and take the first row
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Topic_Perc_Contrib'], ascending=False).head(1)],
                                                axis=0)

    # Reset the index of the sorted DataFrame
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    # Remove unwanted columns if necessary
    sent_topics_sorteddf_mallet = sent_topics_sorteddf_mallet.drop(columns=["Document_No"], errors='ignore')

    # Assign column names for better readability
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

    return sent_topics_sorteddf_mallet

# Function to format topics and their contribution in each document
def format_topics_sentences(ldamodel=None, corpus=None, texts=None): 

    # Verify that parmeters are not None
    if ldamodel is None or corpus is None or texts is None:
        raise ValueError("The LDA model, corpus, and texts must be provided.")
    
    # verify that corpus and texts have the same length
    if len(corpus) != len(texts):
        raise ValueError("The corpus and texts must have the same length.")

    # Initialize a list to store each document's dominant topic and its properties
    records = []

    # Iterate over each document in the corpus
    for i, row_list in tqdm(enumerate(ldamodel[corpus]), desc="iterating through corpus...", total=len(corpus)):

        # Check if the model has per word topics or not to choose the correct element
        row = row_list[0] if ldamodel.per_word_topics else row_list

        # Sort each document's topics by the percentage contribution in descending order
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Extract the dominant topic and its percentage contribution for each document
        for j, (topic_num, prop_topic) in enumerate(row):

            # Only the top topic (dominant topic) is considered
            if j == 0:

                # Get the topic words and weights
                wp = ldamodel.show_topic(topic_num)

                # Join the topic words
                topic_keywords = ", ".join([word for word, prop in wp])

                # Create the records
                record = (int(topic_num), round(prop_topic, 4), topic_keywords)

                # Append the dominant topic and its properties to the list
                records.append(record)

                # Exit the loop after the dominant topic is found
                break

    # Create the DataFrame from the accumulated rows
    sent_topics_df = pd.DataFrame(records, columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])

    # Add the original text of the documents to the DataFrame
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    # Reset the index of the DataFrame for aesthetics and readability
    sent_topics_df = sent_topics_df.reset_index()

    # Rename the columns of the DataFrame for clarity
    sent_topics_df.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    return sent_topics_df





## For visualization

def plot_topic_keywords(lda_model, clean_texts):

    # Extract topics and flatten data
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in clean_texts for w in w_list]
    counter = Counter(data_flat)

    # Initialize empty list to store data
    out = []

    # Iterate over topics and their words to retrieve the weights and
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    # Create DataFrame from collected data
    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)

    # Define colors for each subplot
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    # Iterate over subplots
    for i, ax in enumerate(axes.flatten()):
        # Plot bar chart for word count
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')

        # Create twinx axis for importance
        ax_twin = ax.twinx()

        # Plot bar chart for word importance
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')

        # Set y-axis labels
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030)
        ax.set_ylim(0, 3500)

        # Set title for subplot
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)

        # Hide y-axis ticks
        ax.tick_params(axis='y', left=False)

        # Rotate x-axis labels
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')

        # Add legends
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')

