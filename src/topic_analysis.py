# This package has three function for emergence detection 
# 1. jss(topic1, topic2) --> jss value, which is a semantic similarity measure for two topics
# 2. calculate_similarities(topics_new, topics_old) --> matrix with similarity values, using jss()
# 3. find_emerging_topics(model_output_new, model_output_old) --> emerging/popular for each new topic
# This package has x functions for trend detection: ...
import numpy as np
from scipy.spatial.distance import jensenshannon
import pandas as pd
from sklearn.linear_model import LinearRegression


def laplace_smoothing(model_x, model_output_x, model_y, model_output_y):
    '''
    Returns the smoothed word-topic-distributions for both given models, by throwing the vocabularies of the two models are together
    and changing 0 values to 10**(-12). The other values are copied from the original word-topic-distributions.
    Results in two word-topic-distributions in the form of numpy arrays with k rows and v columns, 
    where v is the size of the combined vocabulary.

    Params: 
     octis.models.LDA.LDA: model_x
     dict: model_output_x (3 entries: 'topics', 'topic-word-matrix' and topic-document-matrix')
     octis.models.LDA.LDA: model_y
     dict: model_output_y (3 entries: 'topics', 'topic-word-matrix' and topic-document-matrix')

    Returns:
     np.array: topic_word_distr_smoothed_x
     np.array: topic_word_distr_smoothed_y
     dict: id2word_joined
    '''
    # Define important variables
    topic_word_distr_x = model_output_x['topic-word-matrix']
    topic_word_distr_y = model_output_y['topic-word-matrix']
    K_x = len(topic_word_distr_x)
    K_y = len(topic_word_distr_y)

    # Iterate over vocabulary to build new topic-word-distributions with the combined vocabulary
    vocab_x = set(model_x.id2word.values())
    vocab_y = set(model_y.id2word.values())
    vocab_joint = sorted(vocab_x | vocab_y) # union of the two sets
    id2word_joined = {i: word for i, word in enumerate(vocab_joint)}

    # Initialize result arrays with correct shape and 10**(-12) as default value
    topic_word_distr_smoothed_x = np.full((K_x, len(vocab_joint)), 10**(-12))
    topic_word_distr_smoothed_y = np.full((K_y, len(vocab_joint)), 10**(-12))

    # Create word to index mappings for fast lookup
    vocab_old_id_x = {word: idx for idx, word in enumerate(model_x.id2word.values())}
    vocab_old_id_y = {word: idx for idx, word in enumerate(model_y.id2word.values())}

    # For each word, check, if word has a value stored in the distribution and write it into the smoothed distribution
    for i, word in enumerate(vocab_joint):
        if word in vocab_old_id_x:
            word_old_index = vocab_old_id_x[word] # lookup old index of the word
            for k_x in range(K_x):
                topic_word_distr_smoothed_x[k_x, i] = topic_word_distr_x[k_x][word_old_index] # save old value of the word in result dict

    # Similarly, smooth distribution of the second model
    for i, word in enumerate(vocab_joint):
        if word in vocab_old_id_y:
            word_old_index = vocab_old_id_y[word] # lookup old index of the word
            for k_y in range(K_y):
                topic_word_distr_smoothed_y[k_y, i] = topic_word_distr_y[k_y][word_old_index] # save old value of the word in result dict

    return topic_word_distr_smoothed_x, topic_word_distr_smoothed_y, id2word_joined


def calculate_similarities(model_new, model_output_new, model_old, model_output_old):
    '''
    Calculates semantic similarites for each of the new topics with each of the old topics using Jensen-Shannon Similarity (JSS),
    which is understood as 1 - JSD (Jensen-Shannon Divergence). JSS is a measure of semantic similarity between two topics, 
    that measures the divergence of their two probability vectors. All values range from 0 to 1.
    Before, the word-topic-distributions are smoothed.
    Returns the results in a matrix with k_new * k_old size.

    Params: 
     octis.models.LDA.LDA: model_new
     dict: model_output_new (3 entries: 'topics', 'topic-word-matrix' and topic-document-matrix')
     octis.models.LDA.LDA: model_old
     dict: model_output_old (3 entries: 'topics', 'topic-word-matrix' and topic-document-matrix')

    Returns:
     np.array(float): similarity_matrix
    '''
    #topics_distr_new = model_output_new['topic-word-matrix']
    #topics_distr_old = model_output_old['topic-word-matrix']

    # Apply Laplace smoothing by adding vocabularies together
    topic_word_distr_new, topic_word_distr_old, id2word_joined = laplace_smoothing(model_new, model_output_new, model_old, model_output_old)

    # Initialize result matrix, where the rows are the new topics and columns are the old topics
    K_new = len(topic_word_distr_new)
    K_old = len(topic_word_distr_old)
    similarity_matrix = np.zeros((K_new, K_old))

    # Iterate over new topics and calculate similarity for each with all of the old topics
    for k_new in np.arange(K_new):
        k_new_distr = topic_word_distr_new[k_new]
        for k_old in np.arange(K_old):
            k_old_distr = topic_word_distr_old[k_old]

            # Calculate jensen shannon similarity and store similarity in result matrix
            jss = 1 - jensenshannon(k_new_distr, k_old_distr)
            similarity_matrix[k_new][k_old] = jss

    return similarity_matrix


def find_emerging_topics(model_new, model_output_new, model_old, model_output_old, pi = 0.3625):
    '''
    Iterates over all topics of a time t and classifies them as either "emerging" or "popular", 
    by using the calculated similaries with the topics found at t-1.
    A new topic is "emerging", if none of its jss-value surpases the threshold --> it is not similar to any of the old topics.
    A new topic is "popular", if at least one of its jss-value surpases the threshold --> it is similar to at least one old topic.
    The threshold is dynamic: It lies between the minimum JSS and the max JSS by a percentage pi.
    pi is set to 0.3625 by default. A smaller pi leads to a more conservative decision, meaning,
    topics have to be even less similar to old topics to be considered new/ "emerging".

    Params: 
     octis.models.LDA.LDA: model_new
     dict: model_output_new (3 entries: 'topics', 'topic-word-matrix' and topic-document-matrix')
     octis.models.LDA.LDA: model_old
     dict: model_output_old (3 entries: 'topics', 'topic-word-matrix' and topic-document-matrix')
     float: pi
     
    Returns:
     dict: result_dict
    '''
    # Define threshold
    similarity_matrix = calculate_similarities(model_new, model_output_new, model_old, model_output_old)
    jss_min = similarity_matrix.min()
    jss_max = similarity_matrix.max()
    threshold = jss_min + pi * (jss_max - jss_min)

    # Initialize result dictionary
    result_dict = {}

    # Iterate over k new topics and define them as either "emerging" or "popular"
    K_new = len(similarity_matrix) #number of rows = number of new topics K

    for k in np.arange(K_new):
        max_jss_of_this_topic = similarity_matrix[k].max()
        if max_jss_of_this_topic > threshold: # it is too similar to another topic seen before -> not new
            t_result = 'popular'
        else: # no other topics are similar above the threshold -> new topic!
            t_result = 'emerging'
        result_dict[k] = t_result

    return result_dict


def assign_s(df, S):
    """
    Assigns a slice value "s" to each row in the given DataFrame according to the number of time slices "S". 
    For example, if t is one year and S is 4, then each s refers to one quarter of the year.
    Returns the dataframe with the additional column "s".
    
    Parameters:
     pd.DataFrame: df
     int: S (size of time interval)
    
    Returns:
     pd.DataFrame: df
    """
    
    # Sort the DataFrame by the date column
    df = df.sort_values(by = 'date').reset_index(drop=True)
    
    # Calculate the time range and duration of each slice
    min_date = df['date'].min()
    max_date = df['date'].max()
    total_days = (max_date - min_date).days + 1
    if total_days < S: # it leads to a division by 0 error -> just set it to 1
        s_size = 1
    else:
        s_size = total_days // S
    
    # Function to determine the slice for a given date
    def get_s(date, min_date, s_size, S):
        num_days = (date - min_date).days
        slice_num = num_days // s_size + 1
        return min(slice_num, S) # make sure s doesn't surpass S, e.g., when 61 days and S=3 -> last day would get s=4
    
    # Apply the function to the DataFrame
    df['s'] = df['date'].apply(get_s, args=(min_date, s_size, S))
    
    return df


def calculate_topic_shares(doc_topic_importances):
    '''
    Calculates topic shares for the topics in the given importance matrix.
    Returns the topic share value for each topic in the matrix as a dictionary.

    Params:
     np.array: topic_doc_importances
    
    Returns:
     dict: topic_shares
    '''
    # Define D_at_s as the number as number of Documents in the given matrix
    D_at_s = len(doc_topic_importances)

    # Define K as the number of topics in the given matrix
    K = len(doc_topic_importances[0])

    # For each topic, count the number of documents where it is "important"
    topics_num_importances = {k: 0 for k in range(K)}
    for doc in doc_topic_importances:
        for k in range(K):
            if doc[k]:
                topics_num_importances[k] += 1
    
    # Divide each absolute frequency by the number of documents at time slice s (D_at_s), to obtain the topic shares
    topic_shares = {k: value/float(D_at_s) for k, value in topics_num_importances.items()}

    return topic_shares


def calculate_topic_growth(model_output, t, S = 4, delta = 0.1):
    '''
    Filters topics importance by returning the topics as important that surpass a threshold delta for the document importance. 
    Mühlroth and Grottke use delta = 0.1.
    Then, calculates topic growth for a given topic, model and number of time slices:
    First, topic share is calculated over the range of possible time slices s. Through these points, a line is fitted.
    Topic growth is defined via the slope of the resulting graph.
    Returns a dictionary that says "growing", if a topic growing, or "declining", if topic share is declining.

    Params:
     dict: model_output (3 entries: 'topics', 'topic-word-matrix' and topic-document-matrix')
     int: t (point in time)
     int: S (size of time intervals)
     float: delta
     
    Returns:
     list: topic_shares_list
    '''
    # Filter topics for document importance (> threshold). Important topics are set to 'True', otherwise 'False.
    topic_doc_distribution = model_output['topic-document-matrix']
    topic_doc_importances = topic_doc_distribution > delta

    # Transpose topic_doc_distribution
    doc_topic_importances = np.transpose(topic_doc_importances)

    # Get date information and assign time slices to dates
    my_path = 'Data/data_for_t' + str(t) + '/doc_id_dates.pkl'
    doc_id_dates = pd.read_pickle(my_path)
    doc_id_dates_s = assign_s(doc_id_dates, S)

    # If there is not enough dates for the calculation, i.e. less unique dates 
    # in the dataframe than requested intervalls S, S has to be set to this number of dates
    if doc_id_dates_s.s.max() < S: # the max s value is the number of unique dates
        S = doc_id_dates_s.s.max()

    # Select relevant batch IDs for all s and calculate topic shares for each part of the matrix
    topic_shares_list = []
    for s in range(1, S+1):

        # Split matrix according to relevant batch IDs
        batch_i_for_this_s = sorted(list(doc_id_dates_s[doc_id_dates_s['s'] == s]['batch_i']))
        doc_topic_importances_s = doc_topic_importances[batch_i_for_this_s, :]
        
        # Check if there are any documents at this time slice s
        if len(doc_topic_importances_s) != 0:
        
            # Calculate Topic shares at s and append to result list
            topic_shares_s = calculate_topic_shares(doc_topic_importances_s)
            topic_shares_list.append([s, topic_shares_s])
    
    return topic_shares_list


def find_trending_topics(model_output, t, S = 4, delta = 0.1):
    '''
    Classifies each topic in the model as "growing" or "declining" by calculating topic growth for each.

    Params:
     dict: model_output (3 entries: 'topics', 'topic-word-matrix' and topic-document-matrix')
     int: t (point in time)
     int: S (size of time intervals)
     float: delta
     
    Returns:
     dict: growth_dict
    '''
    # Calculate topic shares for all topics, for all s in S
    # Result is list of lenght S, where one entry is of the form [s, topic_shares], 
    # where topic_shares is a dictionary of the form {k1: topic_share as float value}
    topic_shares_list = calculate_topic_growth(model_output, t, S = S, delta = delta)
    
    # Extract times (has to be in vertical form (rows) to be a scalar in the regression) and topic shares for regression inputs
    times = np.array([item[0] for item in topic_shares_list]).reshape(-1, 1)
    topics = topic_shares_list[0][1].keys()
    
    # Iterate over topics and calculate linear regression for each topic
    growth_dict = {}

    for topic in topics:
        # Extract the topic's shares (has to be in vertical form (rows) to be a scalar in the regression)
        shares = np.array([item[1][topic] for item in topic_shares_list]).reshape(-1, 1)
       
        # Fit linear regression with times as input (x) and shares as output variable (y), and get the slope of the graph
        model = LinearRegression().fit(times, shares)
        slope = model.coef_[0][0]
        
        # Determine growth status based on the slope
        if slope > 0:
            growth_dict[topic] = 'trending'
        elif slope < 0:
            growth_dict[topic] = 'declining'
        else:
            growth_dict[topic] = 'stable'

    return growth_dict