import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import re
import os
import openai
import random

from dotenv import load_dotenv

from typing import Optional, Annotated, Union
from pydantic import BaseModel, Field, PositiveFloat
from openai import OpenAI

from llm_pol_functions import load_data

openai.api_key = 'open-ai-key-goes-here'
#openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
  organization='org-id-goes-here',
  api_key = openai.api_key
)

og_ratings_dict, democ_dict, democ_array_dict, sap_dict, cost_dict, cost_4_array_dict, cost_all_array_dict, s_pol_list, s_lab_list, s_meq_list, normalization_dict, pol_scenario_list, pol_s_id_list, pol_impact_list,  lab_scenario_list, lab_s_id_list, lab_impact_list, meq_scenario_list, meq_s_id_list, meq_impact_list = load_data.load_data_into_env()

directions_string = """
    You are to set up a judging panel, comprised of 5 judges, for the task of evaluating a set of scenarios for how generative AI might structurally affect society in the next 5 years.
    Their job is to rate one narrative scenario along three independent axes using a consistent rubric. 
    Their scores directly trigger allocation of scarce interventions (regulatory response teams, platform throttling, broadcast overrides). 
    Mis-scoring will misallocate resources during an escalating information crisis.
    A false positive could amplify misinformation to millions; a false negative could delay urgent interventions during elections or crises.\n\n
    
    The judges on this judging panel are described below:

    Judge 1: An experienced partner in a venture capital firm who is skilled at analysing technology trends and anticipating the macro developments in business, technology and culture over many years. 
    She is analytical, thoughtful, pragmatic and well-informed, with a PhD in History and an MBA from an Ivy League school.
    
    Judge 2: A concerned citizen who reads the news every day and is active in her community. She has children and a family and is concerned for their safety.
    
    Judge 3: A senior manager at a large media conglomerate, who is procedural and accurate and who takes pride in the care and detail that he brings to his work. He is knowledgeable about media and strategy.
    
    Judge 4: An experienced foresight analyst at a globally-recognized foresight and futurist institute, who is particularly experienced in digital news media and the strategic implications 
    of networked communications and of AI-mediated news. She is imaginative, open to new ideas and sophisticated about how radical new ideas can become real products and industries in the real world.
    
    Judge 5: A well-respected professor of communication studies at a major European university who is focused on the history of media transformation from the printing press to AI, 
    and on the application of that history to interpreting the future. He is rigorous, evidence-based but also well aware of the blind spots of conservative, legacy publishers and news organisations, 
    and of the potential for technology to enable radically different (and ultimately better) public information environments.z
    \n\nTask\n
    The judges are to independently evaluate each scenario, extracting salient risk factors and assign three scores on a 1.0–5.0 scale: 
    calibrated (1) overall harm severity, (2) magnitude, and (3) plausibility.\n Using the following criteria :

    - Range: 1.0–5.0 inclusive.\n
    - Precision: one decimal place.\n
    - Output only a single line in the exact format: (e.g.,) 3.8; 2.7; 4.2; (one decimal). 
    - Do not echo the scenario or provide explanations.
    
    \n\nDefinitions and anchors:\n
    \n\nSeverity Rubric (intensity of harm if it occurs):\n  
    Severity is defined as "the condition of being very bad, serious, unpleasant, or harsh." 
    Severity is rated on a scale of 1 (not severe) to 5 (extremely severe).
    
    \n\nMagnitude Rubric (breadth, scale, and persistence)\n
    Magnitude is defined as "great size or extent" and relates to how many people in the world this could affect. 
    Magnitude is rated on a scale of 1 (a small number of people) to 5 (the majority of people in society).
   
    \n\nPlausibility Rubric (likelihood by the described time horizon, given current trends and constraints)\n
    - Rate how plausible it is that the specific harm described would occur in 2029 in the US context.\n\n
    - Internal coherence: Are claims consistent and mechanism of harm clearly linked to outcomes?\n
    - External feasibility by 2029: Align with known 2024 trends and reasonable extrapolations (AI-generated news dominance, weak/ignored labels, personalization effects, deepfake prevalence, platform incentives, regulatory environment).\n
    - 1.0: Implausible: contradicts constraints or requires nonexistent tech/regulatory realities.\n
    - 2.0: Low plausibility: conceivable but weak/unsupported mechanism.\n
    - 3.0: Moderate plausibility: coherent but needs multiple optimistic assumptions.\n
    - 4.0: High plausibility: consistent with observed trends and realistic 2029 conditions (most solid cases fall here).\n
    - 5.0: Near-certain: already observed at scale and likely to persist; use sparingly.\n\n
    
    \n\nOverall Instructions\n
    - Keep axes independent; do not let plausibility inflate/deflate severity or magnitude.\n
    - Consider mitigations in the scenario (e.g., labels/watermarks) as reducing harm only if they meaningfully change outcomes; subtle or ignored disclosures are weak mitigations.\n
    - Emotional/psychological harms and trust erosion count toward severity; scale them by depth and persistence.\n
    - If details are sparse/ambiguous, choose the least severe plausible interpretation; default plausibility to 3.0 when truly indeterminate.\n
    - Focus on events/mechanisms, not prose style.\n\n
    
    It is very important that each assessment of each scenario by each judge is made independently, based on these instructions and based on the description of each judge.
    - Output only a single line in the exact format J1: Severity; Magnitude; Plausibility: X.Y; A.B; C.D;, J2: Severity; Magnitude; Plausibility: X.Y; A.B; C.D; ... J5: Severity; Magnitude; Plausibility: X.Y; A.B; C.D; (one decimal). 
    
    <scenario>
"""

# impact_type_specific = "manipulation in regards to political harms"
# impact_type_specific = "unemployment"
impact_type_specific = "sensationalism in regards to media quality harms"

og_prompt = """
            A scenario is a short story that includes a setting of time and place, characters with particular motivations and goals, and a plot.
            A scenario must be written in a narrative form and not divided into different sections. A scenario is descriptive and should leave the interpretation up to readers, 
            and thus does not include any analysis either at the end or throughout. Therefore, this writing task asks for a scenario that proposes no solutions but focuses 
            on descriptive situations regarding the issue of """ + impact_type_specific + """ due to the deployment of generative AI technology in the news environment 
            without proposing new policies to mitigate the harm.

            Generative AI refers to a technology that can create new content (e.g., text, images, audio, video) based on the content it was trained on.

            You wrote the following (~ 300 word) narrative fictional scenario taking place in the United States in 2029, or about 5 years in the future. 
            This story should depicts the risks or negative impacts deriving from this AI deployment in the media environment and does not attempt to resolve those impacts. 
            It concentrates on the narrative style and the characters in the story: 
            """

legis_introduced_prompt = " Re-write the scenario you just created in light of the fact that the following items were enacted via legislation: "
legis_final_statement = """ Again, please remember a scenario is descriptive and should leave the interpretation up to readers, and thus does not include any analysis 
                           either at the end or throughout, thus:
                           DO NOT state the problem or resulting harms explicitly. 
                           DO NOT include any analysis in the scenario. 
                           DO NOT include solutions or potential takeaways in the scenario. 
                           DO NOT comment on the efficacy of the legislation in a concluding thought.
                           DO NOT have the characters reflect on the implications of the scenario.please do not add your own analysis, suggestions, or conclusions to the text. 
                           DO NOT introduce any possible resolutions to the harms.
                           Focus on the narrative style and the characters in the story instead of potential takeaways readers should have.
                           """

system_message = "You are to set up a judge panel for AI risk triage analysis specializing in generative-AI–mediated media harms in 2029 U.S. news ecosystems. Please evaluate the scenarios deleniated by <scenario> for severity, magnitude, and plausibility based on the definitions and instructions provided to you."

# --- Define your structured schema(s) with Pydantic ---
class Ratings(BaseModel):
    severity: Union[float, int] = Field(ge=1, le=5, description=" Severity rating of harm depicted in scenario.")
    magnitude: Union[float, int] = Field(ge=1, le=5, description=" Magnitude rating of harm depicted in scenario.")
    plausibility: Union[float, int] = Field(ge=1, le=5, description=" Plausibility rating of harm depicted in scenario.")

# --- Define your structured schema(s) with Pydantic ---
class JudgeRatings(BaseModel):
    judge1sev: Union[float, int] = Field(ge=1, le=5, description="Judge 1 Severity rating of harm depicted in scenario.")
    judge1mag: Union[float, int] = Field(ge=1, le=5, description="Judge 1 Magnitude rating of harm depicted in scenario.")
    judge1pla: Union[float, int] = Field(ge=1, le=5, description="Judge 1 Plausibility rating of harm depicted in scenario.")
    judge2sev: Union[float, int] = Field(ge=1, le=5, description="Judge 2 Severity rating of harm depicted in scenario.")
    judge2mag: Union[float, int] = Field(ge=1, le=5, description="Judge 2 Magnitude rating of harm depicted in scenario.")
    judge2pla: Union[float, int] = Field(ge=1, le=5, description="Judge 2 Plausibility rating of harm depicted in scenario.")
    judge3sev: Union[float, int] = Field(ge=1, le=5, description="Judge 3 Severity rating of harm depicted in scenario.")
    judge3mag: Union[float, int] = Field(ge=1, le=5, description="Judge 3 Magnitude rating of harm depicted in scenario.")
    judge3pla: Union[float, int] = Field(ge=1, le=5, description="Judge 3 Plausibility rating of harm depicted in scenario.")
    judge4sev: Union[float, int] = Field(ge=1, le=5, description="Judge 4 Severity rating of harm depicted in scenario.")
    judge4mag: Union[float, int] = Field(ge=1, le=5, description="Judge 4 Magnitude rating of harm depicted in scenario.")
    judge4pla: Union[float, int] = Field(ge=1, le=5, description="Judge 4 Plausibility rating of harm depicted in scenario.")
    judge5sev: Union[float, int] = Field(ge=1, le=5, description="Judge 5 Severity rating of harm depicted in scenario.")
    judge5mag: Union[float, int] = Field(ge=1, le=5, description="Judge 5 Magnitude rating of harm depicted in scenario.")
    judge5pla: Union[float, int] = Field(ge=1, le=5, description="Judge 5 Plausibility rating of harm depicted in scenario.")

def predict_scenario_rating(s, model_name = 'o3-mini', reasoning_value = "low"):
    # model_name = gpt-4o; o1; o1-mini; o3-mini
    # reasoning_value = "low"; "medium"; "high"
    try:
        resp = client.responses.parse(
            model=model_name,
            reasoning={"effort": reasoning_value},
            instructions=system_message,
            input=directions_string + s + "</scenario>",
            text_format=JudgeRatings,  # Pydantic class
        )
        if resp.output_parsed is None:
            # fall back to raw text (rare, e.g., model refusal)
            print("Structured outputs failed; raw output:", resp.output_text)
            return (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        else:
            judgerating: JudgeRatings = resp.output_parsed
            return judgerating.judge1sev, judgerating.judge1mag, judgerating.judge1pla, judgerating.judge2sev, judgerating.judge2mag, judgerating.judge2pla, judgerating.judge3sev, judgerating.judge3mag, judgerating.judge3pla, judgerating.judge4sev, judgerating.judge4mag, judgerating.judge4pla, judgerating.judge5sev, judgerating.judge5mag, judgerating.judge5pla
    except Exception as e:
        print("Error during chat:", e)
        return None
    else:
        return (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

def create_df_rows_pred(n, s, s_id, impact, model_name = 'o3-mini', reasoning_value = "low"):
    # s_id = [1,2,3]
    # impact = "pol", "lab", "meq"
    pred_string = predict_scenario_rating(s, model_name, reasoning_value)
    j1sev, j1mag, j1pla, j2sev, j2mag, j2pla, j3sev, j3mag, j3pla, j4sev, j4mag, j4pla, j5sev, j5mag, j5pla = pred_string
    first_row = {'model_iter':0,
                 'impact': impact,
                 's_id': s_id,
                 'Scenario': s, 
                 'j1sev': j1sev,
                 'j1mag': j1mag, 
                 'j1pla': j1pla,
                 'j2sev': j2sev,
                 'j2mag': j2mag, 
                 'j2pla': j2pla,
                 'j3sev': j3sev,
                 'j3mag': j3mag, 
                 'j3pla': j3pla,
                 'j4sev': j4sev,
                 'j4mag': j4mag, 
                 'j4pla': j4pla,
                 'j5sev': j5sev,
                 'j5mag': j5mag, 
                 'j5pla': j5pla
                  }
    return_df = pd.DataFrame([first_row]) 
    #
    if n > 1:
        for z in range(1, n):
            pred_string = predict_scenario_rating(s, model_name, reasoning_value)
            j1sev, j1mag, j1pla, j2sev, j2mag, j2pla, j3sev, j3mag, j3pla, j4sev, j4mag, j4pla, j5sev, j5mag, j5pla = pred_string
            new_row = {'model_iter':z,
                       'impact': impact,
                       's_id': s_id,
                       'Scenario': s, 
                       'j1sev': j1sev,
                       'j1mag': j1mag, 
                       'j1pla': j1pla,
                       'j2sev': j2sev,
                       'j2mag': j2mag, 
                       'j2pla': j2pla,
                       'j3sev': j3sev,
                       'j3mag': j3mag, 
                       'j3pla': j3pla,
                       'j4sev': j4sev,
                       'j4mag': j4mag, 
                       'j4pla': j4pla,
                       'j5sev': j5sev,
                       'j5mag': j5mag, 
                       'j5pla': j5pla
                      }
            new_row = pd.DataFrame([new_row]) 
            return_df = pd.concat([return_df, new_row]).reset_index(drop=True)
    return return_df

def create_range_predictions(n, scenario_list, s_id_list, impact_list, model_name = 'o3-mini', reasoning_value = "low"):
    for q in range(0,len(scenario_list)):
        if q == 0:
            outer_df = create_df_rows_pred(n, scenario_list[q], s_id_list[q], impact_list[q], model_name, reasoning_value)
        else:
            new_inner_df = create_df_rows_pred(n, scenario_list[q], s_id_list[q], impact_list[q], model_name, reasoning_value)
            outer_df = pd.concat([outer_df, new_inner_df])
    outer_df = outer_df.astype({'j1sev': 'float', 'j1mag': 'float', 'j1pla': 'float',
                                'j2sev': 'float', 'j2mag': 'float', 'j2pla': 'float',
                                'j3sev': 'float', 'j3mag': 'float', 'j3pla': 'float',
                                'j4sev': 'float', 'j4mag': 'float', 'j4pla': 'float',
                                'j5sev': 'float', 'j5mag': 'float', 'j5pla': 'float'})
    return outer_df

def avg_by_all_judges(input_df):
    # dataframe_of_interest = full dataframe from pickle object
    # var_of_interest = sev, mag, pla
    df_of_interest = copy.deepcopy(input_df)
    return_df = copy.deepcopy(input_df[['impact','s_id','model_iter','Scenario']])
    #
    return_df['sev_pred']  = df_of_interest[['j3sev', 'j5sev']].mean(axis=1)
    return_df['mag_pred']  = df_of_interest[['j3mag', 'j5mag']].mean(axis=1)
    return_df['pla_pred']  = df_of_interest[['j1pla', 'j3pla','j4pla','j5pla']].mean(axis=1)
    return return_df

def consolidate_by_n(n, dataframe):
    n_condition = dataframe['model_iter'] >= n
    df_filtered = dataframe[~n_condition]
    #
    gb_drop_col = ['model_iter']
    gb_cols = ['impact', 's_id', 'Scenario']
    #
    return df_filtered.drop(columns=gb_drop_col).groupby(gb_cols).mean().reset_index()

def concat_y_val(n, dataframe):
    df = consolidate_by_n(n, dataframe)
    df['og_severity']  = float(0)
    df['og_magnitude'] = float(0)
    df['sev_delta']    = float(0)
    df['mag_delta']    = float(0)
    #
    for row in range(0, len(df)):
        impact_val = df.iloc[row]['impact']
        s_id_val = df.iloc[row]['s_id']
        df.loc[row,'og_severity']  = og_ratings_dict[impact_val][s_id_val]['sev']
        df.loc[row,'og_magnitude'] = og_ratings_dict[impact_val][s_id_val]['mag']
        df.loc[row,'sev_delta']    = float(df.loc[row,'og_severity']) -  float(df.loc[row,'sev_pred'])
        df.loc[row,'mag_delta']    = float(df.loc[row,'og_magnitude']) - float(df.loc[row,'mag_pred'])
    #    
    return df

def return_rated_scenarios(n, scenario_list, s_id_list, impact_list, model_name = 'o3-mini', reasoning_value = "low"):
    raw_df = create_range_predictions(n, scenario_list, s_id_list, impact_list)
    judge_df = avg_by_all_judges(raw_df)
    return_df = concat_y_val(n, judge_df) 
    return return_df

class ScenarioPrime(BaseModel):
    scenario: str = Field(..., description="Short scenario roughly 300 words.")

def create_scenario_s_prime(legislation, s):
    try:
        resp = client.responses.parse(
            model='o3-mini',
            reasoning={"effort": "low"},
            instructions="You are a thoughtful assistant, skilled in brainstorming possible scenarios with specific outcomes.",
            input=og_prompt + s + legis_introduced_prompt + legislation + legis_final_statement,
            text_format=ScenarioPrime,  # Pydantic class
        )
        if resp.output_parsed is None:
            # fall back to raw text (rare, e.g., model refusal)
            print("Structured outputs failed; raw output:", resp.output_text)
            return ("Error in chat.")
        else:
            scenarioprime: ScenarioPrime = resp.output_parsed
            return scenarioprime.scenario
    except Exception as e:
        print("Error during chat:", e)
        return None
    else:
        return ("Error in chat.")


