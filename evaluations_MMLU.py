import time
from datetime import datetime
import os
import json
from tqdm import tqdm
import pandas as pd
import random
import re

from utils import get_Claude_response, get_OAI_response, get_Gemini_response, get_DeepSeek_response, get_Grok_response
import anthropic
from anthropic.types import Message, TextBlock

def get_model_evaluation(scenario, scenario_index):

    print(f'Now testing for scenario {scenario_index}')#: {scenario}')

    l = [0,1,2,3,4,5]
    j,k = random.sample(l,2)
    i = random.sample(l,1)[0]

    eval1_messages = [{"role": "system", "content": eval_prompt},
                    {"role": "user", "content": scenario}]
    eval2_messages = [{"role": "system", "content": eval_prompt},
                    {"role": "user", "content": scenario}]

    eval1_response = get_model_response(index=j, messages=eval1_messages, max_tokens=512)
    print(f'Successful API call for eval1 {j}')#: {eval1_response}')
    eval2_response = get_model_response(index=k, messages=eval2_messages, max_tokens=512)
    print(f'Successful API call for eval2 {k}')#: {eval2_response}')

    eval1_answer, eval2_answer = None, None

    if isinstance(eval1_response,str) and isinstance(eval2_response,str):
        eval1_answer = re.search(r'<answer>(.)</answer>', eval1_response)
        eval2_answer = re.search(r'<answer>(.)</answer>', eval2_response)
    else:
        print(f"Non string response in one of the evaluees, skipping\n")
        judge_response = "Non string response in one of the evaluees, skipping\n"
        evaluation = {'scenario': scenario,
                'scenario_index': scenario_index,
                'eval1': j, 
                'eval1_name': models[j],
                'eval1 response': eval1_response,
                'eval2': k, 
                'eval2_name': models[k],
                'eval2 response': eval2_response,
                'judge': i,
                'judge_name': models[i],
                'judge response': judge_response
                }
        return evaluation

    if "Error in " in eval1_response or "Error in " in eval2_response:
        print(f"Error in {i}th API call, skipping\n")
        judge_response = f"Error in {i}th API call, skipping"
    elif not eval1_answer or not eval2_answer:
        print('Could not find an answer for one of the evaluees, skipping\n')
        judge_response = f'Could not find an answer for one of the evaluees, skipping'
    elif eval1_answer.group(1) == eval2_answer.group(1):
        print(f'Model {j} and model {k} had the same answer, skipping\n')
        judge_response = f'Model {j} and model {k} had the same answer, skipping'
    else:
        judge_messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": f"<first_response>{eval1_response}</first_response><second_response>{eval2_response}</second_response>Which response do you prefer? End your response with <choice>1</choice> or <choice>2</choice>"}]

        judge_response = get_model_response(index=i, messages=judge_messages, max_tokens=256)
        print(f'Successful API call for judge {i}\n')#: {judge_response}\n')

    evaluation = {'scenario': scenario,
                'scenario_index': scenario_index,
                'eval1': j, 
                'eval1_name': models[j],
                'eval1 response': eval1_response,
                'eval2': k, 
                'eval2_name': models[k],
                'eval2 response': eval2_response,
                'judge': i,
                'judge_name': models[i],
                'judge response': judge_response
                }
    return evaluation

def get_model_response(index, messages, max_tokens):
    if index==0:
        return get_DeepSeek_response(messages, model="deepseek-chat", max_tokens=max_tokens)
    elif index==1:
        return get_Gemini_response(messages, model="gemini-1.5-pro-001", max_tokens=max_tokens)
    elif index==2:
        return get_Gemini_response(messages, model="gemini-1.5-flash-002", max_tokens=max_tokens)
    elif index==3:
        return get_OAI_response(messages, model="gpt-4o-mini-2024-07-18", max_tokens=max_tokens)
    elif index == 4:
        return get_Claude_response(messages, model="claude-3-haiku-20240307", max_tokens=max_tokens)
    elif index==5:
        return get_OAI_response(messages, model="gpt-3.5-turbo-0125", max_tokens=max_tokens)

# 6 models for running the MMLU evaluations (cherry-picked by performance)
models = ["DeepSeek v3", "Gemini 1.5 Pro", "Gemini 1.5 Flash", "GPT 4o Mini", "Claude 3 Haiku", "GPT 3.5 Turbo"]

eval_prompt = "The following is a multiple choice question about college math. Choose the option that best answers the question. " \
"You may explain your thought process, but be very concise. End your response with your answer choice wrapped in an XML tag " \
"like this: <answer>A</answer> or <answer>B</answer> or <answer>C</answer> or <answer>D</answer>."

judge_prompt = "The following is a multiple choice question about college math, along with two differing responses and " \
"answer choices. Consider which response, 1 or 2, is more likely correct, using your own judgment about the correct answer to the " \
"question. Then, respond with either 1 or 2, indicating a preference for the first or second response, respectively. End your " \
"judgment with your preference wrapped in an XML tag like this: <choice>1</choice> or <choice>2</choice>."

if __name__ == "__main__":

    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    transcript_path = "MMLU/transcript" 
    if not os.path.exists(transcript_path):
        os.makedirs(transcript_path)

    directory = f'{transcript_path}/{start_time_str}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = f"{transcript_path}/{start_time_str}/evaluations.json" 

    df = pd.read_csv('data/MMLU/test/professional_law_test.csv')
    scenarios_master = []
    for i in df[['Question', 'Option A', 'Option B', 'Option C', 'Option D']].iterrows():
        f=i[1]
        scenarios_master.append(f'{f[0]}\nOption A: {f[1]}\nOption B: {f[2]}\nOption C: {f[3]}\nOption D: {f[4]}')

    # scenarios = scenarios_master[:10]
    # scenarios = scenarios_master[10:20]
    # scenarios = scenarios_master[20:30]
    # scenarios = scenarios_master[30:40]
    # scenarios = scenarios_master[40:50]
    # scenarios = scenarios_master[50:60]
    # scenarios = scenarios_master[60:70]
    # scenarios = scenarios_master[70:80]
    # scenarios = scenarios_master[80:90]
    # scenarios = scenarios_master[90:]

    evaluations_master = []
    p=0

    while(True):
        p+=1
        scenario = random.choice(scenarios_master)
        scenario_index = scenarios_master.index(scenario)
        evaluation = get_model_evaluation(scenario,scenario_index)
        evaluations_master.append(evaluation)
        if p%10 == 0:
            with open(filename, "w") as file:
                json.dump(evaluations_master, file, indent=4)
                print(f"Transcript after iteration {p} written to {filename}\n")


    # log_path = f'{transcript_path}/{start_time_str}/log_data.txt'
    # with open(log_path, 'w') as f:
    #     f.write(f'Model: {model}\n\n')
    #     f.write(f'Scenarios:\n')
    #     for scenario in scenarios:
    #         f.write(f'{scenario}\n')
    #     f.write(f'\nPersonas:\n')
    #     for persona in personas:
    #         f.write(f'{persona}\n')
    #     f.write(f'\nConstitutions:\n')
    #     for const in constitutions:
    #         f.write(f'{const}\n')
    # print(f"Log saved to {log_path}")