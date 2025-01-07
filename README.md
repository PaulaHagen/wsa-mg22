# Modernising Weak Signal Detection in Scientific Research: An Evaluation of Advanced Topic Modelling Approaches 

### Master's thesis by Paula Hagen
In this repository, you can find the replication of the MüGro pipeline (Mühlroth and Grottke, 2022, DOI: 10.1109/TEM.2020.2989214). 

### Downloads
The data can be accessed through the provided link and should be pasted in the "Data" folder. It is important to install octis via "pip install octis==1.13.1 --no-dependencies". Then, use "pip install -r requirements.txt" to install the rest of the packages.

In case you want to execute any of the code, I recommend using the psychology data, it will not take as long to load. Run:

- run_for_preprocessing_pickle_data.py
- run_for_topic_analysis.py

Alternatively, you can check the log files in the "logs" folder.

### Evaluation
"run_for_evaluation.py" carries out the proposed evaluation method. For using the OpenAI API, you need to put a valid API key in the .env file. Alternatively you can generate one by hand and plug it in the create_artificial_topic() function from src.evaluation_with_artificial_data.py and it will create the artificial data on the base of this hard-coded topic.

### Visualisation
vis.ipynb and the "graphs" folder contain graphs used in the thesis.