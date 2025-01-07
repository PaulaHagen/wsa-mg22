import dcl
import pandas as pd
import nltk
# Download nltk stopwords, tokenization and PoS-Taggging
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('popular')
nltk.download('stopwords')


def clean_word(word):
    '''
    Cleans a word of unnecessary white spaces and diacritics and transforms it to lowercase.
    
    Params: 
     str: word

    Returns:
     str: word
    '''
    word_clean = dcl.clean_diacritics(word) # clean of diacritics
    word_clean = word_clean.replace(' ', '') # strip white spaces
    word_clean = word_clean.lower() # transform to lowercase
    return word_clean


def basic_filters(corpus):
    '''
    Cleans all words in "title" and "description" column with the "clean_word" function.
    Filters all words, so there are no digit-only words and no words with special chars in the corpus.
    Keeps only words with 2-30 characters.
    
    Params: 
     pd.DataFrame: data as dataframe

    Returns:
     pd.DataFrame: data as dataframe
    '''
    # Iterate over dataframe
    for index, row in corpus.iterrows():
        title = row['title']
        desc = row['description']

        # Iterate over words in TITLE
        keep_words_title, keep_words_desc = [], []
        for word in title.split():
            
            # Keep word if not digit-only word, 2.) no special chars in word, 3.) relevant word length
            if not word.isdigit() and word.isalnum() and len(word) > 1 and len(word) < 31:
                word_clean = clean_word(word)
                keep_words_title.append(word_clean)
        
        # Combine selected words to new string and select as new title
        corpus.at[index, 'title'] = ' '.join(keep_words_title)
        
        # Iterate over words in DESCRIPTION (same as for titles!)
        for word in desc.split():
            
            # Keep word if 1.) no digit-only word, 2.) no special chars in word, 3.) relevant word length
            if not word.isdigit() and word.isalnum() and len(word) > 1 and len(word) < 31:
                word_clean = clean_word(word)
                keep_words_desc.append(word_clean)
        
        # Combine selected words to new string and select as new description
        corpus.at[index, 'description'] = ' '.join(keep_words_desc)

    return corpus


def tokenize_and_pos_tag(corpus):
    '''
    Tokenizes words in "title" and "description" column and saves them in two new columns.
    Adds PoS Tags to the tokenized words in "title" and "description" column and saves them in two new columns. 
    A word is now represented by a tuple of the word and its tag (both str) and a cell is a list of tuples.
    
    Params: 
     pd.DataFrame: data as dataframe

    Returns:
     pd.DataFrame: data as dataframe
     '''
    # New token columns
    corpus['title_tokens'] = [[] for _ in range(len(corpus))]
    corpus['desc_tokens'] = [[] for _ in range(len(corpus))]

    # Iterate over dataframe
    for index, row in corpus.iterrows():
        title = row['title']
        desc = row['description']

        # Tokenize
        title_tokens = nltk.word_tokenize(title)
        desc_tokens = nltk.word_tokenize(desc)

        # PoS-Tagging
        title_tagged = nltk.pos_tag(title_tokens)
        desc_tagged = nltk.pos_tag(desc_tokens)

        # Write in columns
        corpus.at[index, 'title_tokens'] = title_tagged
        corpus.at[index, 'desc_tokens'] = desc_tagged
    
    return corpus


def filter_pos_tags(corpus):
    '''
    Keeps only tokens with relevant PoS Tags (proper nouns, nouns, verbs, adjectives, numerals).

    Params: 
     pd.DataFrame: data as dataframe

    Returns:
     pd.DataFrame: data as dataframe
     '''
    # Only keep proper nouns, nouns, verbs, adjectives, numerals!
    relevant_pos = ['NN', 'NNP', 'NNPS', 'NNS', # nouns and proper nouns, singular & plural
                    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', # verb forms
                    'CD', 'JJ', 'JJR', 'JJS'] # adjectives and numerals, ordinal & comparative & superlative
    
    # Iterate over dataframe
    for index, row in corpus.iterrows():
        title_tokens = row['title_tokens']
        desc_tokens = row['desc_tokens']
        
        # Iterate over words in TITLE
        keep_pos_title, keep_pos_desc = [], []
        for pair in title_tokens:
            word = pair[0]
            tag = pair[1]
            # Keep word, if relevant PoS-Tag:
            if tag in relevant_pos:
                keep_pos_title.append((word, tag))
        
        # Iterate over words in DESCRIPTION (same as for titles!)
        for pair in desc_tokens:
            word = pair[0]
            tag = pair[1]
            # Keep word, if relevant PoS-Tag:
            if tag in relevant_pos:
                keep_pos_desc.append((word, tag))

        # Write in columns
        corpus.at[index, 'title_tokens'] = keep_pos_title
        corpus.at[index, 'desc_tokens'] = keep_pos_desc

    return corpus


def lemmatize(corpus):
    '''
    Lemmatizes tokens. The "pos" argument of the lemmatizer has the following options: 
    "n" for nouns, "v" for verbs, "a" for adjectives, "r" for adverbs and "s" for satellite adjectives

    Params: 
     pd.DataFrame: data as dataframe

    Returns:
     pd.DataFrame: data as dataframe
     '''
    # Inititalize Lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()

    #  Use lemmatizer argument 'pos': Valid options are "n" for nouns, "v" for verbs, "a" for adjectives, 
    # "r" for adverbs and "s" for satellite adjectives
    tag_type = {'NN': 'n', 'NNP': 'n', 'NNPS': 'n', 'NNS': 'n', 
                'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',
                'CD': 'a', 'JJ': 'a', 'JJR': 'a', 'JJS': 'a'}

    # Iterate over dataframe
    for index, row in corpus.iterrows():
        title_tokens = row['title_tokens']
        desc_tokens = row['desc_tokens']
        
        # Iterate over words in TITLE
        title_lemmas, desc_lemmas = [], []
        for pair in title_tokens:
            word = pair[0]
            tag = pair[1]
            # Lemmatize
            lemmatized_word = lemmatizer.lemmatize(word, pos = tag_type[tag])
            title_lemmas.append((lemmatized_word, tag))
        
        # Iterate over words in DESCRIPTION (same as for titles!)
        for pair in desc_tokens:
            word = pair[0]
            tag = pair[1]
            # Lemmatize
            lemmatized_word = lemmatizer.lemmatize(word, pos = tag_type[tag])
            desc_lemmas.append((lemmatized_word, tag))

        # Write in columns
        corpus.at[index, 'title_tokens'] = title_lemmas
        corpus.at[index, 'desc_tokens'] = desc_lemmas

    return corpus


def filter_stop_words(corpus):
    '''
    Filters the tokens for stop words.

    Params: 
     pd.DataFrame: data as dataframe

    Returns:
     pd.DataFrame: data as dataframe
     '''
    # use only one nltk list for now
    stops = set(nltk.corpus.stopwords.words('english'))

    # Iterate over dataframe
    for index, row in corpus.iterrows():
        title_tokens= row['title_tokens']
        desc_tokens = row['desc_tokens']
        
        # Remove stopwords in all lists
        corpus.at[index, 'title_tokens'] = [(word, tag) for (word, tag) in title_tokens if word not in stops]
        corpus.at[index, 'desc_tokens'] = [(word, tag) for (word, tag) in desc_tokens if word not in stops]

    return corpus


def merge_title_and_description(corpus):
    '''
    Merges title and description tokens in a new column "tokens". Also drops the pos tag information.

    Params: 
     pd.DataFrame: data as dataframe

    Returns:
     pd.DataFrame: data as dataframe
     '''
    # New token column
    corpus['tokens'] = [[] for _ in range(len(corpus))]

    # Iterate over dataframe
    for index, row in corpus.iterrows():
        title_tokens = row['title_tokens']
        desc_tokens = row['desc_tokens']
        tokens = []

        for token_tag_pair in title_tokens:
            tokens.append(token_tag_pair[0])
        
        for token_tag_pair in desc_tokens:
            tokens.append(token_tag_pair[0])

        corpus.at[index, 'tokens'] = tokens

    return corpus


def preprocess(corpus):
    '''
    Preprocesses the data using the other functions above.

    Params: 
     pd.DataFrame: data as dataframe

    Returns:
     pd.DataFrame: data as dataframe
     '''
    corpus = basic_filters(corpus)
    corpus = tokenize_and_pos_tag(corpus)
    corpus = filter_pos_tags(corpus)
    corpus = lemmatize(corpus)
    corpus = filter_stop_words(corpus)
    corpus = merge_title_and_description(corpus)
    return corpus


def calculate_year(date, start_date):
    '''
    Determines the year number of a given date based on the time passed since an earlier date.
    Year is 1 in the first year, so less than 365 days have passed (and so on).

    Params: 
     datetime: date
     datetime: start_date

    Returns:
     int: year number
     '''
    num_days = (date - start_date).days
    return num_days // 365 + 1


def split_in_time_points(corpus):
    '''
    Splits data into many dataframes, one per 365 days. The dataframes are then saved in a dictionary, 
    where the key is the year number and the value is the dataframe.

    Params: 
     pd.DataFrame: data as dataframe

    Returns:
     dict: dictionary of the data split in multiple dataframes
     '''
    # Change the label/date column to datetime data type to make sorting and time intervals possible
    corpus['date'] = pd.to_datetime(corpus['coverDate'])

    # Add year column: "year 1", "year 2" and so on, where one year is a 365 day intervall (not calendar years)
    # Find the earliest date in the DataFrame
    earliest_date = corpus['date'].min()

    # Apply the function to calculate the year for each row
    corpus['year'] = corpus['date'].apply(calculate_year, start_date = earliest_date)

    # Ensure the dataframe is sorted by date
    corpus = corpus.sort_values(by='date')

    # Get unique years in the dataframe
    years = corpus['year'].unique()

    # Initialize an empty dictionary to store the dataframes
    year_dataframes = {}

    # Iterate through each year and accumulate data
    for year in years:
        # Filter the dataframe for the current year
        current_year_data = corpus[corpus['year'] == year]

        # Store the selected data in the dictionary
        year_dataframes[year] = current_year_data

    print(f'Result: number of dataframes/ calculated years/ t: {len(year_dataframes)}')
    return year_dataframes
