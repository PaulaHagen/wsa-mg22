from src.evaluation_with_artificial_data import create_artificial_data, create_artificial_topic
from src.lda_octis import load_octis_data
from datetime import datetime
import pandas as pd
import pickle
import numpy as np


# Set parameters for adding artificial data
T = 20 # total number of t
t_with_artificial = range(2, T+1, 2) # the time t, at which the data should be added --> all even t have artificial data.
percent_artificial = 0.01 # how many percent of the final corpus should be artificial (recommend: 0.5% as topics, that are identified as weak signals are the most relevant topic for 1-2% of the abstracts)

# Generate artificial topic
research_field = 'the sciences' # the research field, for which the data should be generated
artificial_topic = create_artificial_topic(research_field = research_field)
print('Generated artificial_topic: ', artificial_topic)

# Iterate over t and add artificial data at step size 2 --> all even t have artificial data.
for t in t_with_artificial:

    #################################
    ### Make artificial dataframe ###
    #################################

    # Get number of documents in corpus at this t
    dataset_at_t = load_octis_data(t)
    D = len(dataset_at_t.get_corpus())

    # Calculate the number of abstracts to be generated, so that x percent of the final data are artificial
    total_num_art_abstracts = int( (D // (1.0-percent_artificial)) * percent_artificial )

    # Generate dataframe with artificial data
    df_artificial = create_artificial_data(t, artificial_topic, total_num_art_abstracts)


    #############################################
    ### Append data to corpus file for this t ###
    #############################################

    # (watch out: appends, everytime you execute!)
    path_to_corpus = 'Data/data_for_t' + str(t) + '/corpus.tsv'
    with open(path_to_corpus, 'a') as file:
        for i, row in df_artificial.iterrows():
            file.write(row['text'] + '\n')


    ###################################################
    ### Append ID and date data to doc_id_dates.pkl ###
    ###################################################

    # (watch out: appends, everytime you execute! -> Delete old artificial data before)
    # Open old doc_id_dates file
    path_to_id_dates = 'Data/data_for_t' + str(t) + '/doc_id_dates.pkl'
    with open(path_to_id_dates, 'rb') as f:
        doc_id_dates = pickle.load(f)

    # Define indexes for artificial data
    start_index = len(doc_id_dates)
    df_artificial = df_artificial.assign(corpus_index = -99) # corpus index not necessary
    df_artificial['batch_i'] = np.arange(start_index, start_index + len(df_artificial)) # batch_index starting, where existing data ended

    # Merge artificial data with the old and overwrite old file
    merged_doc_id_dates = pd.concat([doc_id_dates, df_artificial[['date', 'batch_i', 'corpus_index']]], ignore_index=True)
    with open(path_to_id_dates, 'wb') as f:
        pickle.dump(merged_doc_id_dates, f)


    ###############################################
    ### Append new vocabulary to vocabulary.txt ###
    ###############################################

    # Get old vocabulary as set
    path_to_vocab = 'Data/data_for_t' + str(t) + '/vocabulary.txt'
    with open(path_to_vocab, 'r') as f:
        lines = f.readlines()
    vocab = set()
    for line in lines:
        vocab.add(line.strip())

    # Add new words to the same set
    for i, row in df_artificial.iterrows():
        for token in row['text'].split():
            vocab.add(token)

    # overwrite the old vocabulary file with the new vocabulary
    with open(path_to_vocab, 'w') as f:
        for word in vocab:
            f.write(word + '\n')

    print(f'Successfully added {total_num_art_abstracts} artificial abstracts at t = {t}')

print(f'Successfully added all artificial abstracts at t = {t_with_artificial}.')