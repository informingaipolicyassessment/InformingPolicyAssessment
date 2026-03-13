import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import re
import os
import openai
from datetime import datetime
import random
import pickle
from llm_pol_functions import open_ai_functions
from llm_pol_functions import load_data

from dotenv import load_dotenv

from typing import Optional, Annotated, Union
from pydantic import BaseModel, Field, PositiveFloat
from openai import OpenAI

openai.api_key = 'open-ai-key-goes-here'
#openai.api_key = os.getenv("OPENAI_API_KEY")

og_ratings_dict, democ_dict, democ_array_dict, sap_dict, cost_dict, cost_4_array_dict, cost_all_array_dict, s_pol_list, s_lab_list, s_meq_list, normalization_dict, pol_scenario_list, pol_s_id_list, pol_impact_list, lab_scenario_list, lab_s_id_list, lab_impact_list, meq_scenario_list, meq_s_id_list, meq_impact_list = load_data.load_data_into_env()
current_directory = os.getcwd()

################################################################################################################
################################################################################################################
####################################### TECHNICAL EVALUATION PIPELINE ##########################################
################################################################################################################
################################################################################################################

def create_initial_vectors(pop_n, sap_dictionary, cost_4_vector):
    chroma_len = len(sap_dictionary.keys())
    shape = (pop_n, chroma_len)
    fill_value = 7
    return_array = np.full(shape, fill_value)
    #
    if (2**sum(cost_4_vector) - 1) < pop_n:
        print("pop_n is too large; not enough unique combinations. Stopping early.")
        return
    #
    counter = 0
    while np.array_equal(return_array[pop_n-1], np.full(chroma_len,7)):
        new_array = np.random.randint(0, 2, size=(1, chroma_len))
        new_array = new_array * cost_4_vector
        if np.any(np.all(return_array == new_array, axis=1)):
            pass
        elif np.all(new_array == 0):
            pass
        else:
            return_array[counter] = new_array
            counter += 1
    #
    return return_array

def transform_array_to_legislation(input_vector, sap_dictionary):
    return_list = []
    for i in range(0, len(input_vector)):
        if int(input_vector[i]) == 1:
            str_val = sap_dictionary[i + 1]['stakeholder'] + " must " + sap_dictionary[i + 1]['action']
            return_list.append(str_val)
    return_str = ""        
    for item in return_list:
        return_str = return_str + item + "\n"
    return return_str

def initalize_population_saps_y_scenarios(pop_n, sap_dictionary, s_list,cost_4_vector):
    column_list = ['input_vector',
                   's1','s2','s3',
                   'sev1','sev2','sev3',
                   'mag1','mag2','mag3',
                   'pla1','pla2','pla3',
                   'og_sev1','og_sev2','og_sev3',
                   'og_mag1','og_mag2','og_mag3',
                  ]
    output_df = pd.DataFrame(None, index=np.arange(pop_n), columns=column_list) 
    #
    init_vectors = create_initial_vectors(pop_n, sap_dictionary,cost_4_vector)
    for row in range(0,len(init_vectors)):
        output_df.loc[row,'input_vector'] = init_vectors[row]
        legistlation_val = transform_array_to_legislation(init_vectors[row], sap_dictionary)
        output_df.loc[row,'s1'] = open_ai_functions.create_scenario_s_prime(legistlation_val, s_list[0])
        output_df.loc[row,'s2'] = open_ai_functions.create_scenario_s_prime(legistlation_val, s_list[1])
        output_df.loc[row,'s3'] = open_ai_functions.create_scenario_s_prime(legistlation_val, s_list[2])
    return output_df

def evaluate_initialized_population(inner_output_df, impact_type_string):
    new_output_df = copy.deepcopy(inner_output_df)
    for row in range(0, len(inner_output_df)):
        scenario_list = []
        for scenario in ['s1','s2','s3']:
            scenario_list.append(inner_output_df.iloc[row][scenario]) 
        ratings_df = open_ai_functions.return_rated_scenarios(1, scenario_list, [0,1,2], [impact_type_string,impact_type_string,impact_type_string])
        ratings_1 = ratings_df[ratings_df['s_id'] == 0]
        ratings_2 = ratings_df[ratings_df['s_id'] == 1]
        ratings_3 = ratings_df[ratings_df['s_id'] == 2]
        new_output_df.loc[row,"sev1"] = float(ratings_1['sev_pred'][0])
        new_output_df.loc[row,"sev2"] = float(ratings_2['sev_pred'][1])
        new_output_df.loc[row,"sev3"] = float(ratings_3['sev_pred'][2])
        new_output_df.loc[row,"mag1"] = float(ratings_1['mag_pred'][0])
        new_output_df.loc[row,"mag2"] = float(ratings_2['mag_pred'][1])
        new_output_df.loc[row,"mag3"] = float(ratings_3['mag_pred'][2])
        new_output_df.loc[row,"pla1"] = float(ratings_1['pla_pred'][0])
        new_output_df.loc[row,"pla2"] = float(ratings_2['pla_pred'][1])
        new_output_df.loc[row,"pla3"] = float(ratings_3['pla_pred'][2])
        new_output_df.loc[row,"og_sev1"] = float(ratings_1['og_severity'][0])
        new_output_df.loc[row,"og_sev2"] = float(ratings_2['og_severity'][1])
        new_output_df.loc[row,"og_sev3"] = float(ratings_3['og_severity'][2])
        new_output_df.loc[row,"og_mag1"] = float(ratings_1['og_magnitude'][0])
        new_output_df.loc[row,"og_mag2"] = float(ratings_2['og_magnitude'][1])
        new_output_df.loc[row,"og_mag3"] = float(ratings_3['og_magnitude'][2])
    return new_output_df

def count_of_plausible(pla1, pla2, pla3):
    list_pla = [pla1, pla2, pla3]
    count_above_3 = int(np.array([1 if v > 3 else 0 for v in list_pla]).sum())
    return count_above_3

def calculate_changes_sev_mag(input_df_raw):
    df = copy.deepcopy(input_df_raw)
    df['sev1_delta'] = df['og_sev1'] - df['sev1']
    df['sev2_delta'] = df['og_sev2'] - df['sev2']
    df['sev3_delta'] = df['og_sev3'] - df['sev3']
    df['mag1_delta'] = df['og_mag1'] - df['mag1']
    df['mag2_delta'] = df['og_mag2'] - df['mag2']
    df['mag3_delta'] = df['og_mag3'] - df['mag3']
    #
    df['sev_avg_delta'] = (df['sev1_delta'] + df['sev2_delta'] + df['sev3_delta'])/3
    df['mag_avg_delta'] = (df['mag1_delta'] + df['mag2_delta'] + df['mag3_delta'])/3
    #
    df['pla_geq_3'] = df.apply(lambda row: count_of_plausible(row['pla1'],row['pla2'],row['pla3']),axis=1)
    #
    cols_to_keep = ['input_vector', 's1','s2','s3', 'sev_avg_delta', 'mag_avg_delta', 'pla_geq_3']
    #
    return df[cols_to_keep]

def add_sum_costs(input_vector_var, impact_type):
    # impact_type = pol, lab, meq
    cost_vector = input_vector_var * cost_all_array_dict[impact_type]
    cost = float(cost_vector.sum())
    return cost

def add_democ_val(input_vector_var, impact_type):
    # impact_type = pol, lab, meq
    # count stipulated
    count_1s = int(input_vector_var.sum())
    # mean priority / agreement score
    mean_prior_agg = float((input_vector_var * democ_array_dict[impact_type]).sum())/count_1s
    return mean_prior_agg

def add_democ_cost_to_df(input_df, impact_type):
    return_df = copy.deepcopy(input_df)
    return_df['cost'] = return_df.apply(lambda row: add_sum_costs(row['input_vector'],impact_type),axis=1)
    return_df['democ_mean_prior_agg'] = return_df.apply(lambda row: add_democ_val(row['input_vector'],impact_type),axis=1)
    return return_df

def normalize_inputs_zscore_df(input_df, impact_string):
    # impact_string = 'pol', 'lab', 'meq'    
    df = copy.deepcopy(input_df)            
    #
    df['sev_mag_avg'] = (df['sev_avg_delta'] + df['mag_avg_delta'])/2
    #
    var_list = ['sev','mag','cost','democ',]
    col_list = ['sev_avg_delta', 'mag_avg_delta', 'cost', 'democ_mean_prior_agg']
    #
    for var in range(0,len(var_list)):
        mean_val = normalization_dict[impact_string]['mean'][var_list[var]]
        std_val = normalization_dict[impact_string]['std'][var_list[var]]
        df["n_" + col_list[var]] = (df[col_list[var]] - mean_val)/std_val
    return df

def optimize_function(input_df_row, 
                      w_sev = 0.5, 
                      w_mag = 0.5, 
                      alpha = 0.34, 
                      beta = 0.33, 
                      gamma = 0.33):
	# Formula:
    # [alpha * (w_sev(n_sev_avg_delta) + w_mag(n_mag_avg_delta))] + -[beta * (n_cost)] + [gamma * n_democ_mean_prior_agg)]
    sev_mag_piece = (w_sev*input_df_row['n_sev_avg_delta']) + (w_mag*input_df_row['n_mag_avg_delta'])
    alpha_piece = alpha * sev_mag_piece
    #
    beta_piece = -1 * beta * input_df_row['n_cost']
    #
    gamma_piece = gamma * input_df_row['n_democ_mean_prior_agg']
    #
    optimal_function = alpha_piece + beta_piece + gamma_piece
    return optimal_function

def check_plausibility(input_df_row):
    if input_df_row['pla_geq_3'] < 3:
        return 1
    else:
        return 0

def check_mag_sev_negative(input_df_row):
    if input_df_row['sev_mag_avg'] < 0:
        return 1
    else:
        return 0

def optimize_columns(input_df, 
                     w_sev = 0.5, 
                     w_mag = 0.5, 
                     alpha = 0.34, 
                     beta = 0.33, 
                     gamma = 0.33):
    return_df = copy.deepcopy(input_df)
    return_df['plausibility_check'] = input_df.apply(lambda row: check_plausibility(row),axis=1)
    return_df['negative_sev_mag_check'] = input_df.apply(lambda row: check_mag_sev_negative(row),axis=1)
    return_df['optimize_function'] = input_df.apply(lambda row: optimize_function(row,w_sev, w_mag, alpha, beta, gamma),axis=1)
    return_df['optimize'] = (return_df['plausibility_check'] * -10_000) + (return_df['negative_sev_mag_check'] * -10_000) + return_df['optimize_function']
    return_df = return_df.sort_values('optimize',ascending=False).reset_index(drop=True)
    #
    column_to_move = return_df.pop("optimize")
    return_df.insert(1, "optimize", column_to_move )
    #
    return return_df

def initialize_and_evalute_population(pop_n, 
                                      sap_dictionary, 
                                      s_list,
                                      cost_4_vector, 
                                      impact_type_string,
                                      w_sev = 0.65, 
                                      w_mag = 0.35, 
                                      alpha = 0.34, 
                                      beta = 0.33, 
                                      gamma = 0.33):
    
    # impact_type_string = 'pol', 'lab', 'meq'
    ##### initialize the population saps and scenarios
    print("starting at: ")
    start_time = datetime.now()
    print(start_time)
    # initialize the sap vectors and pair them with scenarios
    init_saps_df = initalize_population_saps_y_scenarios(pop_n, sap_dictionary, s_list,cost_4_vector)
    # # # # TIME FUNCTIONS # # # #
    print("finished initializing.")
    initializing_time = datetime.now()
    print(initializing_time)
    initializing_minutes = (initializing_time - start_time).seconds/60
    if initializing_minutes > 60:
        initializing_hours = initializing_minutes/60
        print(f"Initializing took {initializing_hours:.2f} hours.") 
    else:
        print(f"Initializing took {initializing_minutes:.2f} minutes.") 
    # # # # TIME FUNCTIONS # # # #
    # have the model rate the scenario primes
    rated_init_saps_df = evaluate_initialized_population(init_saps_df, impact_type_string)
    # # # # TIME FUNCTIONS # # # #
    print("finished rating.")
    rating_time = datetime.now()
    print(rating_time)
    rating_minutes = (rating_time - initializing_time).seconds/60
    if rating_minutes > 60:
        rating_hours = rating_minutes/60
        print(f"Rating took {rating_hours:.2f} hours.") 
    else:
        print(f"Rating took {rating_minutes:.2f} minutes.") 
    # # # # TIME FUNCTIONS # # # #
    # calculate changes between og mag/sev and rated mag/sev
    agg_rated_init_saps_df = calculate_changes_sev_mag(rated_init_saps_df)
    #print("finished calculating changes.")
    # add the democracy values and the cost values to the dataframe based on the sap vector
    raw_full_saps_df = add_democ_cost_to_df(agg_rated_init_saps_df, impact_type_string)
    #print("finished adding democracy and cost values.")
    # normalize the dataframe
    normalized_df = normalize_inputs_zscore_df(raw_full_saps_df,impact_type_string)
    #print("finished normalizing.")
    # add optimization function results to dataframe
    full_df = optimize_columns(normalized_df,
                               w_sev = w_sev,
                               w_mag = w_mag, 
                               alpha = alpha, 
                               beta = beta, 
                               gamma = gamma)
    #print("finished optimizing.")
    rest_time = datetime.now()
    print(rest_time)
    rest_seconds = (rest_time - rating_time).seconds
    #print(f"The rest took {rest_seconds:.2f} seconds.")
    all_minutes = (rest_time - start_time).seconds/60
    if all_minutes > 60:
        all_hours = all_minutes/60
        print(f"Whole thing took {all_hours:.2f} hours.") 
    else:
        print(f"Whole thing took {all_minutes:.2f} minutes.") 
    print("----------")
    return full_df

def add_roulette_wheel_fitness_function(df):
    return_df = copy.deepcopy(df)
    #
    # convert normal dist --> [0,1]
    min_val = df['optimize'].min()
    max_val = df['optimize'].max()
    roulette_ranking = (df['optimize']-min_val)/(max_val-min_val)
    # convert [0,1] -- > [0, population val]
    roulette_ranking = roulette_ranking * 22
    roulette_sum = roulette_ranking.sum()
    roulette_proportion = roulette_ranking/roulette_sum
    return_df['roulette_proportion'] = roulette_proportion
    roulette_wedge = []
    previous_proportion = 0
    for val in roulette_proportion:
        current_mark_off = previous_proportion + val
        roulette_wedge.append(current_mark_off)
        previous_proportion = current_mark_off
    return_df['roulette_mark'] = pd.DataFrame(roulette_wedge)
    column_to_move = return_df.pop("roulette_proportion")
    return_df.insert(2, "roulette_proportion", column_to_move)
    column_to_move = return_df.pop("roulette_mark")
    return_df.insert(3, "roulette_mark", column_to_move)
    return return_df

def select_one_roulette(df):
    val = random.uniform(0, 1)
    min_range = 0
    for z in range(0, len(df)):
        if float(df['roulette_mark'][z]) > val:
            max_range = df['roulette_mark'][z]
            return df['input_vector'][z], min_range, max_range
            break
        min_range = df['roulette_mark'][z]

def select_two_roulette(df, choice_1_min, choice_1_max):
    move_on = False
    while move_on == False:
        val = random.uniform(0, 1)
        if choice_1_min < val < choice_1_max:
            pass
        else:
            move_on = True
    for z in range(0, len(df)):
        if float(df['roulette_mark'][z]) > val:
            return df['input_vector'][z]
            break

def crossover_parents(parent1, parent2, p_c = 0.8, p_mutate = 0.03):
    # first check if the crossover percent chance is reached. if not, return (potentially mutated) parents
    val = random.uniform(0, 1)
    if val > p_c:
        # mutate before sending back original parents
        return_parent1 = copy.deepcopy(parent1)
        return_parent2 = copy.deepcopy(parent2)
        for y in range(0, len(parent1)):
            if random.uniform(0,1) < p_mutate:
                return_parent1[y] = 1 - return_parent1[y]
            else:
                pass
            if random.uniform(0,1) < p_mutate:
                return_parent2[y] = 1 - return_parent2[y]
        return return_parent1, return_parent2
    else:
        point_x = random.randint(0,len(parent1)-1)
        point_y = random.randint(0,len(parent1)-1)
        if point_y >= point_x:
            point_a = point_x
            point_b = point_y
        else:
            point_a = point_y
            point_b = point_x
        arr_1 = parent1[0:point_a]
        arr_2 = parent2[0:point_a]
        #
        arr_1 = np.append(arr_1, parent2[point_a:point_b+1])
        arr_2 = np.append(arr_2, parent1[point_a:point_b+1])
        #
        if (point_b + 1) == len(parent1):
            pass
        else:
            arr_1 = np.append(arr_1, parent1[point_b+1:])
            arr_2 = np.append(arr_2, parent2[point_b+1:])
        # then randomly mutate each value with probability mutate_prob
        for y in range(0, len(arr_1)):
            if random.uniform(0,1) < p_mutate:
                arr_1[y] = 1 - arr_1[y]
            else:
                pass
            if random.uniform(0,1) < p_mutate:
                arr_2[y] = 1 - arr_2[y]
        return arr_1, arr_2

def create_mutated_population(previous_gen_pop_df, 
                              new_pop_n,
                              sap_dictionary, 
                              cost_4_vector,
                              elitism = 3,
                              p_c = 0.8, 
                              p_mutate = 0.03):
    ##### Build shape of new array    
    # length is the number of possible values (aka those with cost < 4)
    chroma_len = len(sap_dictionary.keys())
    shape = (new_pop_n, chroma_len)
    fill_value = 7
    return_array = np.full(shape, fill_value)
    # grab the previously tried vectors so we don't calcuate any rows anew
    prev_return_array = np.vstack(previous_gen_pop_df['input_vector'])
    ##### remove vectors that made net mag/sev worse
    # remove those who failed either plausibility or negative sev/mag test
    prev_df = previous_gen_pop_df[previous_gen_pop_df['optimize'] > -9990]
    ##### bring over the elitism!
    for elite_vec in range(0, elitism):
        return_array[elite_vec] = previous_gen_pop_df['input_vector'][elite_vec]
    ##### add new rows, starting at the first row that wasn't carried over from elitism
    prev_df = add_roulette_wheel_fitness_function(previous_gen_pop_df)
    counter = elitism
    while np.array_equal(return_array[new_pop_n-1], np.full(chroma_len,7)):
        # choose which vectors we're combining
        vector_1, min_range_vec1, max_range_vec1 = select_one_roulette(prev_df)        
        vector_2 = select_two_roulette(prev_df, min_range_vec1, max_range_vec1)
        # combine those vectors
        new_vec1, new_vec2 = crossover_parents(vector_1, vector_2, p_c, p_mutate)
        # multiple by cost_4_vector to make sure we're not doing any cost 4 saps
        new_vec1 = new_vec1 * cost_4_vector
        new_vec2 = new_vec2 * cost_4_vector
        # if vector1 has already been created this time around, pass
        if np.any(np.all(return_array == new_vec1, axis=1)):
            pass
        # if vector is all 0s, pass
        elif np.all(new_vec1 == 0):
            pass
        else:
            return_array[counter] = new_vec1
            counter += 1
        if np.array_equal(return_array[new_pop_n-1], np.full(chroma_len,7)):
            # if vector2 has already been created this time around, pass
            if np.any(np.all(return_array == new_vec2, axis=1)):
                pass
            # if vector is all 0s, pass
            elif np.all(new_vec2 == 0):
                pass
            else:
                return_array[counter] = new_vec2
                counter += 1
        else:
            pass
    return return_array

def mutate_population_saps_y_scenarios(previous_gen_pop_df, 
                                       previously_rated_df,
                                       new_pop_n,
                                       sap_dictionary, 
                                       s_list,
                                       cost_4_vector,
                                       elitism = 3,
                                       p_c = 0.8, 
                                       p_mutate = 0.03):
    column_list = ['input_vector',
                   's1','s2','s3',
                   'sev1','sev2','sev3',
                   'mag1','mag2','mag3',
                   'pla1','pla2','pla3',
                   'og_sev1','og_sev2','og_sev3',
                   'og_mag1','og_mag2','og_mag3',
                  ]
    output_df = pd.DataFrame(None, index=np.arange(new_pop_n), columns=column_list) 
    already_eval_rows = pd.DataFrame(columns=previously_rated_df.columns)
    #
    mutated_vectors = create_mutated_population(previous_gen_pop_df, 
                                                new_pop_n,
                                                sap_dictionary, 
                                                cost_4_vector,
                                                elitism,
                                                p_c, 
                                                p_mutate)
    #
    previously_eval_vectors = np.vstack(previously_rated_df['input_vector'])
    #
    we_are_creating_a_dataframe = 1
    for row in range(0,len(mutated_vectors)):
        # if mutated vector has already been evaluated in a previous iteration, load those values
        if np.any(np.all(previously_eval_vectors == mutated_vectors[row], axis=1)):
            # identify the row in the previous dataframe that rated that vector
            row_of_interest = previously_rated_df[previously_rated_df['input_vector'].apply(lambda x: np.array_equal(x, mutated_vectors[row]))]
            row_of_interest = row_of_interest.reset_index(drop=True)
            #
            if we_are_creating_a_dataframe == 1:
                already_eval_rows = copy.deepcopy(row_of_interest)
                we_are_creating_a_dataframe = 0
            else:
                already_eval_rows = pd.concat([already_eval_rows,row_of_interest]).reset_index(drop=True)
            #
            output_df.loc[row,'input_vector'] = mutated_vectors[row]
            output_df.loc[row,'s1'] = row_of_interest['s1'][0]
            output_df.loc[row,'s2'] = row_of_interest['s2'][0]
            output_df.loc[row,'s3'] = row_of_interest['s3'][0]
        # otherwise, calculate them
        else:        
            output_df.loc[row,'input_vector'] = mutated_vectors[row]
            legistlation_val = transform_array_to_legislation(mutated_vectors[row], sap_dictionary)
            output_df.loc[row,'s1'] = open_ai_functions.create_scenario_s_prime(legistlation_val, s_list[0])
            output_df.loc[row,'s2'] = open_ai_functions.create_scenario_s_prime(legistlation_val, s_list[1])
            output_df.loc[row,'s3'] = open_ai_functions.create_scenario_s_prime(legistlation_val, s_list[2])
    return output_df, already_eval_rows

def evaluate_mutated_population(inner_output_df, previously_rated_df, impact_type_string):
    #
    # get all previously evaluated rows so we don't re-evaluate them
    previously_eval_vectors = np.vstack(previously_rated_df['input_vector'])
    #
    new_output_df = copy.deepcopy(inner_output_df)
    combine_output_df = pd.DataFrame(columns=previously_rated_df.columns)
    #
    for row in range(0, len(inner_output_df)):
        # first check if we've evaluated this before:
        if np.any(np.all(previously_eval_vectors == inner_output_df['input_vector'][row], axis=1)):
            # pass this has already been evaluated
            pass
        else:             
            scenario_list = []
            for scenario in ['s1','s2','s3']:
                scenario_list.append(inner_output_df.iloc[row][scenario]) 
            ratings_df = open_ai_functions.return_rated_scenarios(1, scenario_list, [0,1,2], [impact_type_string,impact_type_string,impact_type_string])
            ratings_1 = ratings_df[ratings_df['s_id'] == 0]
            ratings_2 = ratings_df[ratings_df['s_id'] == 1]
            ratings_3 = ratings_df[ratings_df['s_id'] == 2]
            new_output_df.loc[row,"sev1"] = float(ratings_1['sev_pred'][0])
            new_output_df.loc[row,"sev2"] = float(ratings_2['sev_pred'][1])
            new_output_df.loc[row,"sev3"] = float(ratings_3['sev_pred'][2])
            new_output_df.loc[row,"mag1"] = float(ratings_1['mag_pred'][0])
            new_output_df.loc[row,"mag2"] = float(ratings_2['mag_pred'][1])
            new_output_df.loc[row,"mag3"] = float(ratings_3['mag_pred'][2])
            new_output_df.loc[row,"pla1"] = float(ratings_1['pla_pred'][0])
            new_output_df.loc[row,"pla2"] = float(ratings_2['pla_pred'][1])
            new_output_df.loc[row,"pla3"] = float(ratings_3['pla_pred'][2])
            new_output_df.loc[row,"og_sev1"] = float(ratings_1['og_severity'][0])
            new_output_df.loc[row,"og_sev2"] = float(ratings_2['og_severity'][1])
            new_output_df.loc[row,"og_sev3"] = float(ratings_3['og_severity'][2])
            new_output_df.loc[row,"og_mag1"] = float(ratings_1['og_magnitude'][0])
            new_output_df.loc[row,"og_mag2"] = float(ratings_2['og_magnitude'][1])
            new_output_df.loc[row,"og_mag3"] = float(ratings_3['og_magnitude'][2])
    return new_output_df

def mutate_and_evalute_population(previous_gen_pop_df, 
                                  previously_rated_df,
                                  pop_n, 
                                  sap_dictionary, 
                                  s_list,
                                  cost_4_vector, 
                                  impact_type_string,
                                  elitism = 3,
                                  p_c = 0.8, 
                                  p_mutate = 0.03,
                                  w_sev = 0.65, 
                                  w_mag = 0.35, 
                                  alpha = 0.34, 
                                  beta = 0.33, 
                                  gamma = 0.33):
    # make sure there aren't any duplicate values in our rows that have been evaluated
    previously_rated_df = previously_rated_df.drop_duplicates(subset=['s1','s2','s3']).reset_index(drop=True)
    # impact_type_string = 'pol', 'lab', 'meq'
    ##### initialize the population saps and scenarios
    print("starting at: ")
    start_time = datetime.now()
    print(start_time)
    # initialize the sap vectors and pair them with scenarios
    mutate_saps_df, already_eval_df = mutate_population_saps_y_scenarios(previous_gen_pop_df, 
                                                                         previously_rated_df,
                                                                         pop_n,
                                                                         sap_dictionary, 
                                                                         s_list,
                                                                         cost_4_vector,
                                                                         elitism,
                                                                         p_c, 
                                                                         p_mutate)  
    
    # make sure there aren't any duplicate values in our rows that have been evaluated
    already_eval_df = already_eval_df.drop_duplicates(subset=['s1','s2','s3']).reset_index(drop=True)
    # # # # TIME FUNCTIONS # # # #
    print("finished initializing.")
    initializing_time = datetime.now()
    print(initializing_time)
    initializing_minutes = (initializing_time - start_time).seconds/60
    if initializing_minutes > 60:
        initializing_hours = initializing_minutes/60
        print(f"Initializing took {initializing_hours:.2f} hours.") 
    else:
        print(f"Initializing took {initializing_minutes:.2f} minutes.")
    # # # # TIME FUNCTIONS # # # #    
    # have the model rate the scenario primes
    rated_mutate_saps_df_saps_df = evaluate_mutated_population(mutate_saps_df, previously_rated_df, impact_type_string)
    # # # # TIME FUNCTIONS # # # #
    print("finished rating.")
    rating_time = datetime.now()
    print(rating_time)
    rating_minutes = (rating_time - initializing_time).seconds/60
    if rating_minutes > 60:
        rating_hours = rating_minutes/60
        print(f"Rating took {rating_hours:.2f} hours.") 
    else:
        print(f"Rating took {rating_minutes:.2f} minutes.") 
    # # # # TIME FUNCTIONS # # # #
    # Remove the NaNs (already evaled rows)
    df_to_eval = rated_mutate_saps_df_saps_df[rated_mutate_saps_df_saps_df['sev1'].notna()].reset_index(drop=True)
    # calculate changes between og mag/sev and rated mag/sev
    agg_rated_mutated_saps_df = calculate_changes_sev_mag(df_to_eval)
    #print("finished calculating changes.")
    # add the democracy values and the cost values to the dataframe based on the sap vector
    raw_full_saps_df = add_democ_cost_to_df(agg_rated_mutated_saps_df, impact_type_string)
    #print("finished adding democracy and cost values.")
    # normalize the dataframe
    normalized_df = normalize_inputs_zscore_df(raw_full_saps_df, impact_type_string)
    #print("finished normalizing.")
    # add optimization function results to dataframe
    full_inner_df = optimize_columns(normalized_df,
                                    w_sev = w_sev, 
                                    w_mag = w_mag, 
                                    alpha = alpha, 
                                    beta = beta, 
                                    gamma = gamma)
    # combine newly optimized df with previous rows who made it to this generation
    full_df = pd.concat([full_inner_df,already_eval_df])
    full_df = full_df.sort_values("optimize", ascending=False)
    full_df = full_df.reset_index(drop=True)
    #print("finished optimizing.")
    # # # # TIME FUNCTIONS # # # #
    rest_time = datetime.now()
    print(rest_time)
    rest_seconds = (rest_time - rating_time).seconds
    #print(f"The rest took {rest_seconds:.2f} seconds.")
    all_minutes = (rest_time - start_time).seconds/60
    if all_minutes > 60:
        all_hours = all_minutes/60
        print(f"Whole thing took {all_hours:.2f} hours.") 
    else:
        print(f"Whole thing took {all_minutes:.2f} minutes.") 
    # # # # TIME FUNCTIONS # # # #
    print("----------")
    return full_df

def run_genetic_algorithm(string_dict_name,
                          storage_dictionary_gen,
                          storage_dictionary_prev,
                          pop_n, 
                          sap_dictionary, 
                          s_list,
                          cost_4_vector, 
                          impact_type_string,
                          min_generations = 10,
                          max_generations = 100,
                          stall_max_generations = 5,
                          elitism = 3,
                          p_c = 0.8, 
                          p_mutate = 0.03,
                          w_sev = 0.65, 
                          w_mag = 0.35, 
                          alpha = 0.34, 
                          beta = 0.33, 
                          gamma = 0.33):    
    # FIRST: Initialize population!
    print("Starting first ever initialization: ")
    start_time_1 = datetime.now()
    print(start_time_1)
    
    # initialize propulation
    init_df = initialize_and_evalute_population(pop_n, 
                                                sap_dictionary, 
                                                s_list,
                                                cost_4_vector, 
                                                impact_type_string,
                                                w_sev, 
                                                w_mag, 
                                                alpha, 
                                                beta,
                                                gamma)
    # store this init as key 0 in the df you pass
    storage_dictionary_gen[0] = init_df
    
    # SECOND: first mutation:
    print("Starting first generation mutation: ")
    start_time_2 = datetime.now()
    print(start_time_2)
    
    # mutate the first generation
    mutated_df = mutate_and_evalute_population(init_df, 
                                               init_df,
                                               pop_n, 
                                               sap_dictionary, 
                                               s_list,
                                               cost_4_vector, 
                                               impact_type_string,
                                               elitism,
                                               p_c, 
                                               p_mutate,
                                               w_sev, 
                                               w_mag, 
                                               alpha, 
                                               beta, 
                                               gamma)
    
    # store this init as key 1 in the df you pass
    storage_dictionary_gen[1] = mutated_df
    storage_dictionary_prev[1] = init_df
    
    # identify the best optimized value for identifying if we're stalling/plateauing generation to generation
    best_optimize = float(mutated_df.sort_values("optimize").reset_index(drop=True)['optimize'][0])
    
    # THIRD: Now mutate for however many generations you want, or until the top value stays the same three times in a row:
    prev_rated_df = pd.concat([init_df, mutated_df]).sort_values("optimize", ascending=False)
    prev_rated_df = prev_rated_df.drop_duplicates(subset=['s1','s2','s3']).reset_index(drop=True)
    prev_rated_df = prev_rated_df.reset_index(drop=True)
    
    # best optimize becomes old for comparison sake
    old_mutated_df = copy.deepcopy(mutated_df)
    old_best_optimize = copy.deepcopy(best_optimize)
    
    stopping_criteria = "not_met"
    counter = 2
    stall_generations = 0
    
    while stopping_criteria != "met":
        # # # # # TIME FUNCTIONS # # # # #
        print("----------")
        print("Starting generation mutation #", str(counter))
        start_time_3 = datetime.now()
        print(start_time_3)
        # # # # # TIME FUNCTIONS # # # # #
        new_mutated_df = mutate_and_evalute_population(old_mutated_df, 
                                                       prev_rated_df,
                                                       pop_n, 
                                                       sap_dictionary, 
                                                       s_list,
                                                       cost_4_vector, 
                                                       impact_type_string,
                                                       elitism,
                                                       p_c, 
                                                       p_mutate,
                                                       w_sev, 
                                                       w_mag, 
                                                       alpha, 
                                                       beta, 
                                                       gamma)
        # store df in dict you passed
        storage_dictionary_gen[counter] = new_mutated_df
        storage_dictionary_prev[counter] = prev_rated_df
        with open(current_directory + '/data/pickle_outputs/' + string_dict_name + '_gen.pkl', 'wb') as file:
            pickle.dump(storage_dictionary_gen, file)
        with open(current_directory + '/data/pickle_outputs/' + string_dict_name + '_prev.pkl', 'wb') as file:
            pickle.dump(storage_dictionary_prev, file)
        # find the best optimized value from the fitness function
        best_optimize = float(new_mutated_df.sort_values("optimize",ascending=False).reset_index(drop=True)['optimize'][0])
        # update prev_rated_df to have current df and all old dfs
        prev_rated_df = pd.concat([new_mutated_df, prev_rated_df]).sort_values("optimize", ascending=False)
        prev_rated_df = prev_rated_df.drop_duplicates(subset=['s1','s2','s3']).reset_index(drop=True)
        prev_rated_df = prev_rated_df.reset_index(drop=True)
        # update "old" df for next round to be df we just created
        old_mutated_df = copy.deepcopy(new_mutated_df)
        # moving onto next generation. let's see if we should, or stop
        counter += 1
        if best_optimize == old_best_optimize:
            print("best:", str(best_optimize))
            print("best old:", str(old_best_optimize))
            stall_generations += 1
            print("adding a stall generation counter; total:", str(stall_generations))
        else:
            print("best:", str(best_optimize))
            print("best old:", str(old_best_optimize))
            stall_generations = 0
            print("NOT adding a stall generation counter; total down to zero:", str(stall_generations))
        old_best_optimize = copy.deepcopy(best_optimize)
        if counter < min_generations:
            print("Minimum generations not met. Continuing.")
        else:
            if counter > max_generations:
                stopping_criteria = "met"
                print("Max generations met! Stopping.")
            elif stall_generations >= stall_max_generations:
                stopping_criteria = "met"
                print("Max stall generations met! Stopping.")
            else:
                print("Neither max generations nor stall generations met. Continuing.")
                print("----------")
    return new_mutated_df









