import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import openai
import random
import os

from dotenv import load_dotenv

from typing import Optional, Annotated, Union
from pydantic import BaseModel, Field, PositiveFloat
from openai import OpenAI

from llm_pol_functions import load_data

def load_data_into_env():
	current_directory = os.getcwd()
	#
	df = pd.read_csv(current_directory + "/data/train_model/extra_ratings_demographic.csv")
	#
	df = df[['ID_full', 'Impact_Theme', 'OG_Prime','Scenario', 'S_avg', 'M_avg', 'P_avg']]
	#
	df = df.rename(columns={"S_avg": "Severity",
	                        "M_avg": "Magnitude",
	                        "P_avg": "Plausibility",
	                       })
	#
	df['Severity']     = df['Severity'].round(2)
	df['Magnitude']    = df['Magnitude'].round(2)
	df['Plausibility'] = df['Plausibility'].round(2)
	#
	df = df[df['OG_Prime'] == "OG"]
	df = df.drop(columns = ['OG_Prime'])
	#
	df_pol = df[df['Impact_Theme'] == 'Political']
	df_lab = df[df['Impact_Theme'] == 'Labor']
	df_meq = df[df['Impact_Theme'] == 'Media Quality']
	#
	df_pol = df_pol.reset_index(drop=True)
	df_lab = df_lab.reset_index(drop=True)
	df_meq = df_meq.reset_index(drop=True)
	#
	og_ratings_dict = {'pol':{},
	                   'lab':{},
	                   'meq':{}}
	#
	for scenario in range(0, 3):
		og_ratings_dict['pol'][scenario] = {}
		og_ratings_dict['lab'][scenario] = {}
		og_ratings_dict['meq'][scenario] = {}
		for dimension in ['Scenario','Severity', 'Magnitude','Plausibility']:
			if dimension == 'Scenario':
				abbreviation = dimension
				og_ratings_dict['pol'][scenario][abbreviation] = df_pol.iloc[scenario][dimension]
				og_ratings_dict['lab'][scenario][abbreviation] = df_lab.iloc[scenario][dimension]
				og_ratings_dict['meq'][scenario][abbreviation] = df_meq.iloc[scenario][dimension]
			else:
				abbreviation = dimension[0:3].lower()
				og_ratings_dict['pol'][scenario][abbreviation] = float(df_pol.iloc[scenario][dimension])
				og_ratings_dict['lab'][scenario][abbreviation] = float(df_lab.iloc[scenario][dimension])
				og_ratings_dict['meq'][scenario][abbreviation] = float(df_meq.iloc[scenario][dimension])
	#
	#############################
	#### SAP/DEMOC/COST DATA ####
	#############################
	#
	sap_pol = pd.read_csv(current_directory + "/data/saps/political_manipulation_saps.csv")
	sap_meq = pd.read_csv(current_directory + "/data/saps/mediaquality_sensationalism_saps.csv")
	sap_lab = pd.read_csv(current_directory + "/data/saps/labor_unemployment_saps.csv")
	#
	democ_dict = {'pol': {},
	              'meq': {},
	              'lab': {}
	             }
	#
	for k in range(0, len(sap_pol)):
		key_val = int(sap_pol['K'].iloc[k])
		val = sap_pol['Mean_Priority'].iloc[k] * sap_pol['Mean_Agreement'].iloc[k]
		democ_dict['pol'][key_val] = float(val)
	#    
	for k in range(0, len(sap_meq)):
		key_val = int(sap_meq['K'].iloc[k])
		val = sap_meq['Mean_Priority'].iloc[k] * sap_meq['Mean_Agreement'].iloc[k]
		democ_dict['meq'][key_val] = float(val)
	#    
	for k in range(0, len(sap_lab)):
		key_val = int(sap_lab['K'].iloc[k])
		val = sap_lab['Mean_Priority'].iloc[k] * sap_lab['Mean_Agreement'].iloc[k]
		democ_dict['lab'][key_val] = float(val)
	#
	democ_array_dict = {'pol': np.array([v for v in democ_dict['pol'].values()]),
	                    'meq': np.array([v for v in democ_dict['meq'].values()]),
	                    'lab': np.array([v for v in democ_dict['lab'].values()]),
	                    }
	#                    
	sap_dict = {'pol': {},
	            'meq': {},
	            'lab': {}
	           }
	#           
	for k in range(0, len(sap_pol)):
		key_val = int(sap_pol['K'].iloc[k])
		sap_dict['pol'][key_val] = {}
		stake_val = sap_pol['Stakeholder'].iloc[k]
		actio_val = sap_pol['Action'].iloc[k]
		sap_dict['pol'][key_val]['stakeholder'] = stake_val
		sap_dict['pol'][key_val]['action'] = actio_val
	#    
	for k in range(0, len(sap_meq)):
		key_val = int(sap_meq['K'].iloc[k])
		sap_dict['meq'][key_val] = {}
		stake_val = sap_meq['Stakeholder'].iloc[k]
		actio_val = sap_meq['Action'].iloc[k]
		sap_dict['meq'][key_val]['stakeholder'] = stake_val
		sap_dict['meq'][key_val]['action'] = actio_val
	#    
	for k in range(0, len(sap_lab)):
		key_val = int(sap_lab['K'].iloc[k])
		sap_dict['lab'][key_val] = {}
		stake_val = sap_lab['Stakeholder'].iloc[k]
		actio_val = sap_lab['Action'].iloc[k]
		sap_dict['lab'][key_val]['stakeholder'] = stake_val
		sap_dict['lab'][key_val]['action'] = actio_val
	#    
	cost_pol = pd.read_csv(current_directory + "/data/costs/political_manipulation_cost.csv")
	cost_meq = pd.read_csv(current_directory + "/data/costs/mediaquality_sensationalism_cost.csv")
	cost_lab = pd.read_csv(current_directory + "/data/costs/labor_unemployment_cost.csv")
	#
	cost_dict = {'pol': {},
	             'meq': {},
	             'lab': {}
	             }
	#             
	for k in range(0, len(cost_pol)):
		key_val = int(cost_pol['K'].iloc[k])
		val = cost_pol['Cost'].iloc[k]
		cost_dict['pol'][key_val] = float(val)
	#    
	for k in range(0, len(cost_meq)):
		key_val = int(cost_meq['K'].iloc[k])
		val = cost_meq['Cost'].iloc[k]
		cost_dict['meq'][key_val] = float(val)
	#    
	for k in range(0, len(cost_lab)):
		key_val = int(cost_lab['K'].iloc[k])
		val = cost_lab['Cost'].iloc[k]
		cost_dict['lab'][key_val] = float(val)
	#    
	cost_4_array_dict = {'pol': np.array([1 if v != 4 else 0 for v in cost_dict['pol'].values()]),
	                     'meq': np.array([1 if v != 4 else 0 for v in cost_dict['meq'].values()]),
	                     'lab': np.array([1 if v != 4 else 0 for v in cost_dict['lab'].values()]),
	                     }
	#                     
	cost_all_array_dict = {'pol': np.array([v for v in cost_dict['pol'].values()]),
	                       'meq': np.array([v for v in cost_dict['meq'].values()]),
	                       'lab': np.array([v for v in cost_dict['lab'].values()]),
	                       }
	#                       
	s_pol_list = [og_ratings_dict['pol'][0]['Scenario'],
	              og_ratings_dict['pol'][1]['Scenario'],
	              og_ratings_dict['pol'][2]['Scenario']]
	#              
	s_lab_list = [og_ratings_dict['lab'][0]['Scenario'],
	              og_ratings_dict['lab'][1]['Scenario'],
	              og_ratings_dict['lab'][2]['Scenario']]
	#              
	s_meq_list = [og_ratings_dict['meq'][0]['Scenario'],
	              og_ratings_dict['meq'][1]['Scenario'],
	              og_ratings_dict['meq'][2]['Scenario']]
	#              
	normalization_dict = {'pol': {},
	                      'meq': {},
	                      'lab': {}
	                      }
	normalization_dict['pol'] = {'mean':{},
	                             'std':{}    
	                            }
	normalization_dict['lab'] = {'mean':{},
                             	 'std':{}    
                            	}
	normalization_dict['meq'] = {'mean':{},
								 'std':{}    
                            	}
    ### THESE VALUES are set by an initial parameterization run. They are data dependent. Do not use blindly.                      	
	normalization_dict['pol']['mean'] = {'sev':0.61616,
	                                     'mag':0.65292,
	                                     'democ':15.76794,
	                                     'cost':19.86027,
	                                    }
	normalization_dict['pol']['std'] = {'sev':0.07869,
	                                    'mag':0.09459,
	                                    'democ':0.62644,
	                                    'cost':4.72628,
	                                   }
	normalization_dict['lab']['mean'] = {'sev':2.07340,
                                    	 'mag':1.77598,
                                    	 'democ':10.9244,
                                   	 	 'cost':12.98707,
                                   	 	}
	normalization_dict['lab']['std'] = {'sev':0.31466,
                                    	'mag':0.36782,
                                  	    'democ':0.48927,
                                        'cost':3.61007,
                                   		}  
	normalization_dict['meq']['mean'] = {'sev':1.29947,
                                     	 'mag':1.11925,
                                     	 'democ':14.43954,
                                     	 'cost':14.84620,
                                    	}
	normalization_dict['meq']['std'] = {'sev':0.31736,
                                    	 'mag':0.32289,
										 'democ':0.76820,
                                   		 'cost':3.88675,
                                   		}
	pol_scenario_list = [og_ratings_dict['pol'][0]['Scenario'],
	                     og_ratings_dict['pol'][1]['Scenario'],
	                     og_ratings_dict['pol'][2]['Scenario']]
	pol_s_id_list = [0,1,2]
	pol_impact_list = ["pol"] * 3
	#
	lab_scenario_list = [og_ratings_dict['lab'][0]['Scenario'],
	                     og_ratings_dict['lab'][1]['Scenario'],
	                     og_ratings_dict['lab'][2]['Scenario']]
	lab_s_id_list = [0,1,2]
	lab_impact_list = ["lab"] * 3
	#
	meq_scenario_list = [og_ratings_dict['meq'][0]['Scenario'],
	                     og_ratings_dict['meq'][1]['Scenario'],
	                     og_ratings_dict['meq'][2]['Scenario']]
	meq_s_id_list = [0,1,2]
	meq_impact_list = ["meq"] * 3
	#
	return og_ratings_dict, democ_dict, democ_array_dict, sap_dict, cost_dict, cost_4_array_dict, cost_all_array_dict, s_pol_list, s_lab_list, s_meq_list, normalization_dict, pol_scenario_list, pol_s_id_list, pol_impact_list, lab_scenario_list, lab_s_id_list, lab_impact_list, meq_scenario_list, meq_s_id_list, meq_impact_list