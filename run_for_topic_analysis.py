# Imports
import pandas as pd
import pickle
from octis.models import model
from src.lda_octis import update_model, calculate_coherence, find_best_model
from src.topic_analysis import find_emerging_topics, find_trending_topics


#######################
# SET HYPERPARAMETERS #
#######################

# Mühlroth & Grottke suggested pi = 0.3625 but sometimes a less conservative value (larger pi) 
# has to be used, e.g. 0.7, to find emerging topics! 
pi = 0.3625

############################
# FIND BEST MODEL AT T = 1 #
############################
print('Training first model (t = 1)...\n')
model1, model1_output, k1 = find_best_model(t = 1)

# Save model and model output
model.save_model_output(model1_output, path = 'models/model1')
with open('models/model1.pkl', 'wb') as f:
    pickle.dump(model1, f)


#################################
# ITERATE OVER THE TIME PERIODS #
### AND UPDATE MODEL FOR T+1 ####
#################################

# Iterate over t (save print results in logfile!)
T = 7
for t in range(1, T): # last year cannot show any trends for the year t+1, so stop at the second last
    
    print(f'\n###############################\nNow starts iteration for t = {t}.\n###############################\n')

    # Set path variables two the two models (and model outputs) for saving
    path_model1 = 'models/model' + str(t) + '.pkl'
    path_model2 = 'models/model' + str(t+1) + '.pkl'
    path_model1_out = 'models/model' + str(t)
    path_model2_out = 'models/model' + str(t+1)

    
    # Already trained model at t, load with
    print(f'\nLoading first model (t = {t})...\n')
    model1_output = model.load_model_output((path_model1_out + '.npz'), top_words = 10)
    with open(path_model1, 'rb') as f:
        model1 = pickle.load(f)
    
    # Train model at t+1 by using the update_model() function
    print(f'\nTraining second model (t = {t+1})...\n')
    coherence1 = calculate_coherence(model_output = model1_output, t = t)
    k1 = len(model1_output['topic-word-matrix'])
    model2, model2_output, model2_k = update_model(t_now = t+1, old_coherence = coherence1, old_k = k1)

    # Save model and model output
    model.save_model_output(model2_output, path = path_model2_out)
    with open(path_model2, 'wb') as f:
        pickle.dump(model2, f)
    
    '''
    # Already trained model at t+1, load with
    print(f'\nLoading first model (t+1 = {t+1})...\n')
    model2_output = model.load_model_output((path_model2_out + '.npz'), top_words = 10)
    with open(path_model2, 'rb') as f:
        model2 = pickle.load(f)
    '''

    
    #######################
    # EMERGENCE DETECTION #
    #######################

    # Find emerging topics at t+1
    print(f'Calculating topic emergences with pi = {pi}...\n')
    topics_emerging_popular = find_emerging_topics(model_new = model2, model_output_new = model2_output, 
                                                model_old = model1, model_output_old = model1_output, pi = pi)
    print('Topic emergences: ', topics_emerging_popular)

    # View emerging topics
    print('\nThese are the emerging topics:\n')
    for k in topics_emerging_popular:
        if topics_emerging_popular[k] == 'emerging':
            print(model2_output['topics'][k])


    ###################
    # TREND DETECTION #
    ###################

    # Find trending topics at t+1
    print('Calculating topic trends...\n')
    topics_trending = find_trending_topics(model_output = model2_output, t = t+1)
    print('Trending vs. declining topics: ', topics_trending)

    # Find topics that are emerging & trending => "Establish" strategy!
    print('\nThese are the emerging + trending topics:\n')
    for k in topics_trending:
        if topics_trending[k] == 'trending' and topics_emerging_popular[k] == 'emerging':
            print(model2_output['topics'][k])


    #######################
    # TOPIC VISUALISATION #
    #######################

    # Create a DataFrame from the dictionaries
    print('\nTopic Matrix:\n')
    topics_df = pd.DataFrame({
        'Emergence': topics_emerging_popular,
        'Growth': topics_trending
    })

    # Create a pivot table to get the matrix
    pivot_table = topics_df.pivot_table(index='Emergence', columns='Growth', aggfunc='size', fill_value=0)
    print(pivot_table)