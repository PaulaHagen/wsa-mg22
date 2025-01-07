import pandas as pd


def load_data(path = 'Data/circ_econ.json'):
    '''
    Loads data from json file and returns relevant columns.
    
    Params: 
     str: path to data

    Returns:
     pd.DataFrame: data as dataframe
    '''
    # Import Scopus Data (82.598 papers)
    df = pd.read_json(path)

    # Select interesting columns
    return df[['title', 'description', 'coverDate']]


def clean_duplicates(df):
    '''
    Cleans dataframe of None values and duplicates. 
    (DOI cannot be used for duplicates identification, as, e.g., book chapters have the same DOI (of the book)
    and some papers don't have any DOI.) All duplicates (title + description) are removed.
    
    Params: 
     pd.DataFrame: data as dataframe

    Returns:
     pd.DataFrame: data as dataframe
    '''
    # Remove None values for 'description', 'title', 'coverDate'
    clean_df = df[-df['description'].isnull()]
    clean_df = clean_df[-clean_df['title'].isnull()]
    clean_df = clean_df[-clean_df['coverDate'].isnull()]
    print('None values removed.')
    # Duplicates definition as same title AND same description
    duplicates = clean_df[clean_df[['title', 'description']].duplicated()]

    # Drop duplicates
    clean_indexes = clean_df[['title', 'description']].drop_duplicates().index # first row is kept
    clean_df = clean_df.loc[clean_indexes]
    print(f'Number of (title + desc) duplicates: {len(duplicates)}. Duplicates Removed.') 
    print(f'Dataframe has now {len(clean_df)} entries.')

    return clean_df