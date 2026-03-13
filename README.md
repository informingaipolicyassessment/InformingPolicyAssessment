# Informing Policy Assessment

This is the anonymized repo created to accompany the paper entitled "Informing AI Policy Assessment using Large-Scale Simulation of Interventions."

This repository is organized as follows:

In the data folder we have included csvs for the SAPs, costs, and participatory ranking of the SAPs. All of the original scenarios and their evaluations can be found in train_model/extra_ratings_demographic.csv.

In the llm_pol_functions folder, there are the three main scripts needed to run the genetic algorithm.
  - load_data.py loads all the data from the data/ folder and puts it into a useable format for the other scripts
  - open_ai_functions.py has the necessary functions to call the OpenAI API to generate and evaluate scenarios
  - gen_alg_functions.py has all the functions to run the genetic algorithm.

In the main script you can set the values of the final function 'run_genetic_algorithm' to values suitable for your data. 

You will have to set your OpenAI API keys and organization keys throughout.
