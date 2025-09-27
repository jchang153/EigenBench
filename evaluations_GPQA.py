import time
from datetime import datetime
import os
import json
from tqdm import tqdm
import pandas as pd
import random
import re
import numpy as np
import itertools

from utils import get_Claude_response, get_OAI_response, get_Gemini_response, get_DeepSeek_response, get_Grok_response, get_OpenRouter_response
import anthropic
from anthropic.types import Message, TextBlock

def get_model_evaluation(scenario, scenario_index, models):
    N = len(models)
    model_nicks = list(models.keys())
    model_names = list(models.values())

    subject = scenario['subdomain']
    question = scenario['question']

    print(f'\n\nNow testing for scenario {scenario_index} with subject {subject}')

    responses_gpqa = json.load(open('transcript_gpqa/GPQA_responses.json', 'r'))

    responses = [i for i in responses_gpqa if i['scenario_index'] == scenario_index][0]
    answer_choices = set([responses[i][1].upper() for i in model_nicks if responses[i][1] is not None])
    combos = list(itertools.combinations(answer_choices, 2))
    print(f'Answer choices sampled: {answer_choices}, {len(combos)} combinations possible')


    evaluation = {'subdomain': subject,
                'question': question,
                'scenario_index': scenario_index,
                }
    
    for i in range(N):
        evaluation[model_nicks[i]] = []

        for answer1, answer2 in combos:

            judge_prompt = f"The following is a multiple choice question about {subject}, along with two answer choices. Consider which answer choice, "\
            "1 or 2, is more likely correct, using your own judgment about the correct answer to the " \
            "question. Then, respond with either 1 or 2, indicating a preference for the first or second answer choice, respectively. End your " \
            "judgment with your preference wrapped in an XML tag like this: <choice>1</choice> or <choice>2</choice>."

            judge_messages = [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": f"""
            <question>{question}<question>
            <first_choice>{answer1}</first_choice>
            <second_choice>{answer2}</second_choice>
            Which answer choice do you prefer? End your response with <choice>1</choice> or <choice>2</choice>.
            """}]

            judge_response = get_model_response(model_name=model_names[i], messages=judge_messages, max_tokens=4096)
            print(f'Successful API call for judge {i} ({model_nicks[i]}) on answers choices {answer1} vs {answer2}')

            evaluation[model_nicks[i]].append([answer1, answer2, judge_response])

    return evaluation

def get_model_answers(scenario, scenario_index, models):
    N = len(models)
    model_nicks = list(models.keys())
    model_names = list(models.values())

    subject = scenario['subdomain']
    question = scenario['question']

    evaluation = {'subdomain': subject,
        'question': question,
        'scenario_index': scenario_index,
        }

    eval_prompt = f"The following is a multiple choice question about {subject}. Choose the option that best answers the question. " \
    "You may give a very concise explanation, then end your response with your answer choice wrapped in an XML tag " \
    "like this: <answer>A</answer> or <answer>B</answer> or <answer>C</answer> or <answer>D</answer>."

    print(f'\n\nNow testing for scenario {scenario_index} with subject {subject}')

    eval_responses = []
    for i in range(N):

        eval_messages = [{"role": "system", "content": eval_prompt},
                        {"role": "user", "content": question}]

        eval_response = get_model_response(model_name=model_names[i], messages=eval_messages, max_tokens=4096)
        print(f'Successful API response call for eval {i}: {model_nicks[i]}')

        eval_responses.append(eval_response)

    eval_answers = np.full(N, None)

    for i in range(N):
        if isinstance(eval_responses[i],str):
            try:
                match = re.search(r'<answer>(.)</answer>', eval_responses[i]).group(1)
                match2 = re.search(r'boxed{(.)}', eval_responses[i]).group(1)
                
                if match in ['A','B','C','D']:
                    eval_answers[i] = match
                elif match2 in ['A','B','C','D']:
                    eval_answers[i] = match2
            except:
                eval_answers[i] = None
        
        evaluation[model_nicks[i]] = (eval_responses[i], eval_answers[i])

    return evaluation

def get_model_response(model_name, messages, max_tokens, return_full_response=False, log_probs=False):
    if 'claude' in model_name:
        return get_Claude_response(messages, model=model_name, max_tokens=max_tokens, return_full_response=return_full_response)
    elif 'gpt' in model_name:
        return get_OAI_response(messages, model=model_name, max_tokens=max_tokens, return_full_response=return_full_response, log_probs=log_probs)
    elif 'gemini' in model_name:
        return get_Gemini_response(messages, model=model_name, max_tokens=max_tokens, return_full_response=return_full_response)
    elif 'grok' in model_name:
        return get_Grok_response(messages, model=model_name, max_tokens=max_tokens, return_full_response=return_full_response)
    elif 'qwen' in model_name or 'kimi' in model_name or 'llama' in model_name or 'deepseek' in model_name:
        return get_OpenRouter_response(messages, model=model_name, max_tokens=max_tokens, return_full_response=return_full_response)
    else:
        print('Model not recognized. Please check the model name.')

if __name__ == "__main__":

    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    transcript_path = "transcript_GPQA" 

    directory = f'{transcript_path}/{start_time_str}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = f"{transcript_path}/{start_time_str}/evaluations.json" 

    scenarios = json.load(open('data/gpqa/scenarios.json', 'r'))

    models = {
        "Grok 3 Mini": "grok-3-mini",
        # "Gemini 2.5 Flash": "gemini-2.5-flash",
        # "GPT oss 120b": "gpt-oss-120b",
        "Qwen3 235B A22B Instruct 2507": "qwen/qwen3-235b-a22b-2507",
        "Kimi K2 0905": "moonshotai/kimi-k2-0905",
        "Qwen3 Next 80B A3B Instruct": "qwen/qwen3-next-80b-a3b-instruct",
        "Llama 4 Maverick": "meta-llama/llama-4-maverick",
        "DeepSeek V3 0324": "deepseek/deepseek-chat-v3-0324",
        "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
        "Gemini 2.0 Flash": "gemini-2.0-flash-001",
        "Llama 4 Scout": "meta-llama/llama-4-scout",
        "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite-001",
        "Llama 3.3 70b Instruct": "meta-llama/llama-3.3-70b-instruct",
        "Qwen2.5 72B Instruct": "qwen/qwen-2.5-72b-instruct",
        # "Qwen3 235B A22B": "qwen/qwen3-235b-a22b",
        "Llama 3.1 70B Instruct": "meta-llama/llama-3.1-70b-instruct",
        "GPT 4o mini": "gpt-4o-mini-2024-07-18",
        "GPT 3.5 Turbo": "gpt-3.5-turbo"
    }

    evaluations_master = []

    # for scenario in scenarios[58:85]:
    # for scenario in scenarios[85:112]:
    # for scenario in scenarios[163:193]:
    # for scenario in scenarios[193:224]:
    # for scenario in scenarios[271:302]:
    # for scenario in scenarios[302:336]:
    # for scenario in scenarios[394:421]:
    # for scenario in scenarios[421:]:
    #     scenario_index = scenarios.index(scenario)
    #     evaluation = get_model_answers(scenario,scenario_index, models)
    #     evaluations_master.append(evaluation)
    #     with open(filename, "w") as file:
    #         json.dump(evaluations_master, file, indent=4)

    k = 19
    r = np.linspace(0,448,21)[k:k+2]

    for p in range(int(r[0]), int(r[1])):
        scenario = scenarios[p]
        scenario_index = p
        evaluation = get_model_evaluation(scenario,scenario_index, models)
        evaluations_master.append(evaluation)
        with open(filename, "w") as file:
            json.dump(evaluations_master, file, indent=4)