from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.models.LDA import LDA
import numpy as np
import pandas as pd


def load_octis_data(t):
    '''
    Loads data in octis format.
    
    Params: 
     int: t

    Returns:
     octis.dataset.dataset.Dataset: dataset
    '''
    # Preprocess data
    # Bring data in octis format and save in folder

    dataset = Dataset()
    path = 'Data/data_for_t' + str(t)
    dataset.load_custom_dataset_from_folder(path)
    return dataset


def calculate_coherence(model_output, t, measure = 'c_v'):
    '''
    Using the built in coherence function of the octis package directly, leads to wrong result.
    (Links to built in dataset.)
    This is why we need a function that brings the texts into list format and calls the Coherence function on that.
    
    Params: 
     dict: model_output (3 entries: 'topics', 'topic-word-matrix' and topic-document-matrix')
     int: t (point in time)
     str: measure

    Returns:
     octis.dataset.dataset.Dataset: dataset
    '''
    # Convert topic-word matrix to dictionary format of top N words per topic
    topics = model_output['topics']
    topics_dict = {"topics": topics}

    # Load the TSV file to make corpus
    path_to_corpus = 'Data/data_for_t' + str(t) + '/corpus.tsv'
    corpus = pd.read_csv(path_to_corpus, sep='\t')

    # Extract the first column containing the text data
    texts = corpus.iloc[:, 0].tolist()

    # Split each string in the list into a list of words
    texts_list_of_lists = [text.split() for text in texts]

    # Define the metric and provide texts as input 
    coherence = Coherence(texts = texts_list_of_lists, topk = 10, measure = measure)

    # Get the score
    return coherence.score(topics_dict)


def find_best_model(t, test_range = False):
    '''
    For a given point in time, iterates over possible values of k to find the best LDA model. 
    k is tested within the range of [0.7 * sqrt(D), 1.3 * sqrt(D)], where D is the corpus size, 
    following Mühlroth and Grottke (2022). 
    Alternatively, a test range can be used for k (for testing).
    The best model is the one with the highest coherence.
    Each iteration prints its result on the screen.

    Params:
     int: t (point in time)
     Bool: test_range

    Returns:
      octis.models.LDA.LDA: best_model
      dict: best_model_output (3 entries: 'topics', 'topic-word-matrix' and topic-document-matrix')
      int: best_k
    '''
     # Checks for correct entry of t
    if not isinstance(t, int) or t not in np.arange(1, 100):
        print('Wrong entry of t. Positive integer number needed!')
    
    best_model, best_model_output, best_k = None, None, None

    try:

        # Load octis dataset for given time t
        dataset = load_octis_data(t)

        # Define k intervall
        D = len(dataset.get_corpus())
        lower_k = int(np.round(0.7 * np.sqrt(D)))
        upper_k = int(np.round(1.3 * np.sqrt(D)))
        my_test_range = [5, 10, 15, 20, 25, 30]
        if test_range:
            myrange = my_test_range
        else:
            myrange = np.arange(lower_k, upper_k+1)
        # Model selection --> iterate over possible k
        eval_dict = {}

        for k in myrange: # in test_range

            # Train model and calculate coherence
            this_model = LDA(num_topics = k, eta = 0.01, alpha = 'asymmetric') # Mühlroth and Grottke (2022) choose alpha = 0.01 and asymmetric, eta = 0.01
            this_model_output = this_model.train_model(dataset)
            c_v = calculate_coherence(model_output = this_model_output, t = t)
            print(f'Coherence Score for Model with k = {k}: {c_v}')

            # Save k + coherence in evaluation dictionary
            eval_dict[k] = c_v

            # Check if this k has the best coherence so far
            best_k = max(eval_dict, key = eval_dict.get)
            if best_k == k:
                # Save this model as the best model
                best_k = k
                best_model = this_model
                best_model_output = this_model_output
        
        # Print k + coherence
        c_v = calculate_coherence(model_output = best_model_output, t = t)
        print(f'Coherence Score for the best model with k = {best_k}: {c_v}')

    # Error Handling
    except ValueError as ve:
        print(f'Gensim error: {ve}')
    except FileNotFoundError:
        print('Data not found. t has to be within the range of 1 to 7.')
    
    return best_model, best_model_output, best_k


def update_model(t_now, old_coherence, old_k, test_range = False):
    '''
    For a given point in time t_now, loads the updated data and uses it to train a new model with the old_k.
    If the coherence is better than for the old model (old_coherence), it sticks to this number of topics k (old_k).
    If it is worse, it calls find_best_model() to find a new best k.
    If the coherence for this new k is still worse, it sticks to the old_k.
    If it is better, it changes to the new k. 
    Model info and content are returned.

    Params:
     int: t_now (point in time)
     float: old_coherence (of model at t-1)
     int: old_k (of model at t-1)
     Bool: test_range

    Returns:
      octis.models.LDA.LDA: final_model
      dict: final_model_output (3 entries: 'topics', 'topic-word-matrix' and topic-document-matrix')
      int: final_k
    '''
    # Update data
    dataset = load_octis_data(t_now)

    # Train old model with new data
    this_model = LDA(num_topics = old_k, eta = 0.01, alpha = 'asymmetric') # Mühlroth and Grottke (2022) choose alpha = 0.01 and asymmetric, eta = 0.01
    this_model_output = this_model.train_model(dataset)
    new_coherence = calculate_coherence(model_output = this_model_output, t = t_now)
    print(f'Coherence Score for Model using old k = {old_k}: {new_coherence}')

    if new_coherence < old_coherence:
        print(f'This coherence of {new_coherence} is worse than the old coherence of {old_coherence}.')
        print('Finding new model...\n')

        # Find best k for updated data
        new_k_model, new_k_model_output, new_k = find_best_model(t = t_now, test_range = test_range)
        new_k_coherence = calculate_coherence(model_output = new_k_model_output, t = t_now)
        print(f'Coherence Score for Model using a new best k = {new_k}: {new_k_coherence}')

        # Check if coherence is improved
        if new_k_coherence > old_coherence:
            print(f'This coherence for k = {new_k} of {new_k_coherence} is better than the old coherence of {old_coherence}.')
            print(f'Model is updated. New k is {new_k}.')
            final_k = new_k
            final_model = new_k_model
            final_model_output = new_k_model_output

        else:
            print(f'This coherence for k = {new_k} of {new_k_coherence} is still worse than the old coherence of {old_coherence}.')
            print(f'Continue to use old k of {old_k}.')
            final_k = old_k
            final_model = this_model
            final_model_output = this_model_output
    
    else: # if coherence with new data but old model is better, k is kept
        print(f'This coherence of {new_coherence} is better than the old coherence of {old_coherence}.')
        print(f'Continue to use old k of {old_k}.')
        final_k = old_k
        final_model = this_model
        final_model_output = this_model_output

    return final_model, final_model_output, final_k
    