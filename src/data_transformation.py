import dcl
import pandas as pd
import nltk
from collections import defaultdict
import numpy as np
import os

def add_bigrams(corpus):
    '''
    Adds bigrams to the tokens of the corpus.

    Params: 
     pd.DataFrame: data as dataframe

    Returns:
     pd.DataFrame: data as dataframe
    '''
    # Create bigrams
    for index, row in corpus.iterrows():
        tokens = row['tokens']
        bigrams_list = list(nltk.bigrams(tokens))
        bigrams = ['_'.join(bigram) for bigram in bigrams_list] # Convert bigrams to word1_word2 format
        corpus.at[index, 'tokens'] = tokens + bigrams
    
    return corpus


def filter_doc_extreme_values(doc, word_doc_freq, num_total_docs, min_doc_freq=3, max_doc_freq_ratio=0.9):
    '''
    Filters tokens based on given document frequency. (Helper function)

    Params: 
     list (str): doc
     int: word_doc_freq
     int: num_total_docs
     int: min_doc_freq
     float: max_doc_freq_ratio

    Returns:
     list: filtered_tokens
    '''
    max_doc_freq = max_doc_freq_ratio * num_total_docs
    filtered_tokens = [token for token in doc
                         if min_doc_freq <= word_doc_freq[token] <= max_doc_freq]
    return filtered_tokens


def filter_extreme_values(corpus, min_doc_freq=3, max_doc_freq_ratio=0.9):
    '''
    Filters tokens for extreme values: Words, that appear in more than 3 and less than 90% of the documents, are saved in a list.

    Params: 
     pd.DataFrame: corpus

    Returns:
     list (list(str)): documents
    '''
    # Filter for abstract length. Minimum 20 words per abstract!
    #corpus = corpus[corpus['description'].apply(lambda x: len(x.split()) >= 20)]

    # Make list of documents and a list of all tokens
    documents, all_tokens = [], []

    for index, row in corpus.iterrows():
        tokens = row['tokens']
        this_doc = []
        
        for token in tokens:
            this_doc.append(token)
            all_tokens.append(token)

        documents.append(this_doc)

    # Calculate document frequencies for tokens (unigrams and bigrams are combined)
    word_doc_freq = defaultdict(int)

    for doc in documents:
        unique_tokens = set(doc)
        for token in unique_tokens:
            word_doc_freq[token] += 1
            # "word_doc_freq" shows, in how many documents a word appears (NOT how often)

    # Transform documents to include only filtered unigrams and bigrams
    docs_with_filtered_tokens = []
    for doc in documents:
        filtered_tokens_and_bigrams = filter_doc_extreme_values(doc, word_doc_freq, len(documents), min_doc_freq, max_doc_freq_ratio)
        docs_with_filtered_tokens.append(filtered_tokens_and_bigrams)

    return docs_with_filtered_tokens


def save_doc_ids_dates(base_folder_path, batch_corpus, year, save_to_folder = True):
    '''
    Iterates over a batch corpus and saves the date information for the given IDs.

    Params:
     pd.DataFrame: batch_corpus
     int: year
    '''
    i = 0
    doc_id_dates = []

    # Iterate over corpus and save corpus_index, batch_i and date in a dictionary
    for index, row in batch_corpus.iterrows():
        id_date_dict = {'corpus_index': index, 'batch_i': i, 'date': row['date']}
        doc_id_dates.append(id_date_dict)
        i+=1

    doc_id_dates_df = pd.DataFrame(doc_id_dates)

    if save_to_folder:
        # Create the directory if it doesn't exist
        folder_path = base_folder_path + 'data_for_t' + str(year)
        os.makedirs(folder_path, exist_ok=True)
        my_path = folder_path + '/doc_id_dates.pkl'
        
        # Save to directory
        doc_id_dates_df.to_pickle(my_path)

    return doc_id_dates_df


def transform_in_octis_format(base_folder_path, document_list, t, save_to_folder = True):
    '''
    Brings a given document list in the correct format for octis.
    For octis, the input data is not a DTM but a folder, with a corpus.tsv and a vocabulary.txt, where one line = one word.
    Filters for minimum document length of 30 tokens. Saves corpus and vocabulary in file format, one folder per t.
    Also saves doc IDs + dates in a separate document each.

    Params: 
     list: list of str (document)

    Returns:
     pd.DataFrame: data as dataframe
     list (str): vocabulary
    '''
    # Write document list into dataframe
    documents = [' '.join(doc) for doc in document_list]
    documents_df = pd.DataFrame(documents)

    '''
    # Add train/test/validation partition column -> Skip this step, as it makes it impossible to track the index of the documents, 
    # which is necessary for later topic trend detection
    proportions = [0.8, 0.1, 0.1]
    labels = ['train', 'test', 'val']
    documents_df['partition'] = np.random.choice(labels, size = len(documents_df), p = proportions)
    
    # Name columns -> Doesn't work, because the column name is interpreted as data by the model (even though, authors tell you to name the column)
    documents_df.columns = ['document']#, 'partition']
    '''

    # Create vocabulary using filtered docs
    vocabulary = set()
    for index, row in documents_df.iterrows():
        for word in row.iloc[0].split():
            if word:
                vocabulary.add(word)

    # Save corpus and vocabulary as files
    if save_to_folder:
        # Construct the path to the files for this t
        folder_path = base_folder_path + 'data_for_t' + str(t)

        # Create the directory if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Write corpus to tsv file
        path_to_corpus = os.path.join(folder_path, 'corpus.tsv')
        documents_df.to_csv(path_to_corpus, sep='\t', index=False, header=False, lineterminator = False) 

        # Write vocabulary to txt file
        path_to_vocabulary = os.path.join(folder_path, 'vocabulary.txt')
        with open(path_to_vocabulary, 'w') as text_file:
            for token in vocabulary:
                text_file.write(token + '\n')

    return documents_df, vocabulary