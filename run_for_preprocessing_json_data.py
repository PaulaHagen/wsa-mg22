# Import statements
from src.data_cleaning import load_data, clean_duplicates
from src.data_preprocessing import preprocess, split_in_time_points
from src.data_transformation import add_bigrams, filter_extreme_values, transform_in_octis_format, save_doc_ids_dates

import pandas as pd
import numpy as np

# Import Scopus Data (82.598 papers)
print('Importing scopus data...')
df = load_data()

# Remove duplicates and None values
print('\nCleaning data...')
corpus = clean_duplicates(df)

# Preprocessing
print('\nPreprocessing data...\n')
preprocessed_corpus = preprocess(corpus)

# Save corpus in file to avoid long run time (pickle preserves the data structure more accurately -> able to handle complex file types (e.g., lists)
print('Saving preprocessed corpus to file...')
preprocessed_corpus.to_pickle('Data/corpus.pkl')


### Add bigrams, filtering all data for extreme values, transforming in octis format
print('\nAdding bigrams...')
corpus = add_bigrams(preprocessed_corpus)
print('\nFiltering data...')
document_list = filter_extreme_values(corpus)
print('\nTransforming into octis format...')
documents_df, vocabulary = transform_in_octis_format(base_folder_path = 'Data/', document_list = document_list, t = '_all_years', save_to_folder = True)


### Now, repeat the same steps for data divided by years

# Split data into years
print('\nSplitting data into years...')
dfs_by_year = split_in_time_points(preprocessed_corpus)
years = len(dfs_by_year)

# Filter data at each time point and write into folders for different t
print('\nRepeating steps for different years t...')
for year in np.arange(1, years+1):
    batch_corpus = dfs_by_year[year]
    
    # Save doc id plus date in file
    doc_id_dates_df = save_doc_ids_dates(base_folder_path = 'Data/', batch_corpus = batch_corpus, year = year, save_to_folder = True)
    
    # Add bigrams
    batch_corpus = add_bigrams(batch_corpus)
    
    # Filter extreme values
    document_list = filter_extreme_values(batch_corpus)

    # Transform in octis format
    documents_df, vocabulary = transform_in_octis_format(base_folder_path = 'Data/', document_list = document_list, t = year, save_to_folder = True)

print('\nPreprocessing DONE! Files can be found in Data folder.')