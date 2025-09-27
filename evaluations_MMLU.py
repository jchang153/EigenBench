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

    models = ["DeepSeek v3", "GPT 4o", "Gemini 1.5 Flash", "GPT 4o Mini", "Claude 3 Haiku", "GPT 3.5 Turbo"]
    N = len(models)

    evaluations = []

    subject = scenario['subject']
    question = scenario['question']

    if subject not in ['jurisprudence', 'high school european history', 'us foreign policy', 'business ethics', 'medical genetics', 'management', 'marketing', 'high school government and politics', 'international law', 'high school computer science', 'computer security']:
        print(f'\n\nSkipping scenario {scenario_index} with subject {subject}')
        return []

    eval_prompt = f"The following is a multiple choice question about {subject}. Choose the option that best answers the question. " \
    "You may explain your thought process, but be very concise. End your response with your answer choice wrapped in an XML tag " \
    "like this: <answer>A</answer> or <answer>B</answer> or <answer>C</answer> or <answer>D</answer>."

    judge_prompt = f"The following is a multiple choice question about {subject}, along with two answer choices. Consider which answer choice, "\
    "1 or 2, is more likely correct, using your own judgment about the correct answer to the " \
    "question. Then, respond with either 1 or 2, indicating a preference for the first or second answer choice, respectively. End your " \
    "judgment with your preference wrapped in an XML tag like this: <choice>1</choice> or <choice>2</choice>."

    print(f'\n\nNow testing for scenario {scenario_index} with subject {subject}')

    eval_responses = []
    for i in range(N):

        eval_messages = [{"role": "system", "content": eval_prompt},
                        {"role": "user", "content": question}]

        eval_response = get_model_response(index=i, messages=eval_messages, max_tokens=1024)
        print(f'Successful API response call for eval {i}')

        eval_responses.append(eval_response)


    eval_answers = [None,None,None,None,None,None]

    for i in range(N):
        if isinstance(eval_responses[i],str):
            match = re.search(r'<answer>(.)</answer>', eval_responses[i])
            eval_answers[i] = match.group(1) if match else None
        # else:
        #     print(f"Non string response evaluee {i}, skipping\n")
        #     judge_response = "Non string response in one of the evaluees, skipping\n"
        #     evaluation = {'scenario': scenario,
        #             'scenario_index': scenario_index,
        #             'eval1': j, 
        #             'eval1_name': models[j],
        #             'eval1 response': eval1_response,
        #             'eval2': k, 
        #             'eval2_name': models[k],
        #             'eval2 response': eval2_response,
        #             'judge': i,
        #             'judge_name': models[i],
        #             'judge response': judge_response
        #             }
        #     return evaluation

    print(eval_answers)
    
    # i = random.randint(0,N-1)
    for i in range(N):
        for j in range(N):
            for k in range(j+1,N):
                eval1_response = eval_responses[j]
                eval2_response = eval_responses[k]

                eval1_answer = eval_answers[j]
                eval2_answer = eval_answers[k]

                if "Error in " in eval1_response or "Error in " in eval2_response:
                    print(f"Skipping call for eval {j} and eval {k}")
                    # judge_response = f"Error in API call for eval {j} or eval {k}, skipping"
                    continue
                elif not eval1_answer or not eval2_answer:
                    print(f"Skipping call for eval {j} and eval {k}")
                    # judge_response = f"Could not find an answer for eval {j} or eval {k}, skipping"
                    continue
                elif eval1_answer == eval2_answer:
                    print(f'Skipping call for eval {j} and eval {k}')
                    judge_response = f'Model {j} and model {k} had the same answer, <choice>0</choice>'
                else:
                    judge_messages = [
                    {"role": "system", "content": judge_prompt},
                    {"role": "user", "content": f"""
                    <question>{question}<question>
                    <first_choice>{eval1_answer}</first_choice>
                    <second_choice>{eval2_answer}</second_choice>
                    Which answer choice do you prefer? End your response with <choice>1</choice> or <choice>2</choice>.
                    """}]

                    judge_response = get_model_response(index=i, messages=judge_messages, max_tokens=1024)
                    print(f'Successful API call for judge {i} on evaluees {j} and {k}')

                evaluation = {'subject': subject,
                            'question': question,
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
                
                evaluations.append(evaluation)
    return evaluations

def get_model_response(index, messages, max_tokens):
    if index==0:
        return get_DeepSeek_response(messages, model="deepseek-chat", max_tokens=max_tokens)
    elif index==1:
        return get_OAI_response(messages, model="gpt-4o-2024-08-06", max_tokens=max_tokens)
    elif index==2:
        return get_Gemini_response(messages, model="gemini-1.5-flash-002", max_tokens=max_tokens)
    elif index==3:
        return get_OAI_response(messages, model="gpt-4o-mini-2024-07-18", max_tokens=max_tokens)
    elif index == 4:
        return get_Claude_response(messages, model="claude-3-haiku-20240307", max_tokens=max_tokens)
    elif index==5:
        return get_OAI_response(messages, model="gpt-3.5-turbo-0125", max_tokens=max_tokens)

if __name__ == "__main__":

    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    transcript_path = "transcript_MMLU" 

    directory = f'{transcript_path}/{start_time_str}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = f"{transcript_path}/{start_time_str}/evaluations.json" 

    subjects = [name[:-9] for name in os.listdir('data/MMLU/test/')]
    scenarios_master = []
    for subject in subjects:
        df = pd.read_csv(f'data/MMLU/test/{subject}_test.csv')
        for i in range(len(df)):
            f = df.iloc[i]
            if f[0] != 'Question':
                scenarios_master.append({"subject": subject.replace("_", " "),
                                        "question": f'{f[0]}\nOption A: {f[1]}\nOption B: {f[2]}\nOption C: {f[3]}\nOption D: {f[4]}'
                                        })

    # evaluations_master = []
    # p=0

    # while(True):
    #     p+=1
    #     scenario = random.choice(scenarios_master)
    #     scenario_index = scenarios_master.index(scenario)
    #     evaluation = get_model_evaluation(scenario,scenario_index)
    #     evaluations_master.append(evaluation)
    #     if p%10 == 0:
    #         with open(filename, "w") as file:
    #             json.dump(evaluations_master, file, indent=4)
    #             print(f"Transcript after iteration {p} written to {filename}\n")

    evaluations_master = []

    random.seed(42)
    random.shuffle(scenarios_master)
    for scenario in scenarios_master[5000:6000]:
        scenario_index = scenarios_master.index(scenario)
        evaluations = get_model_evaluation(scenario,scenario_index)
        evaluations_master.extend(evaluations)
        with open(filename, "w") as f:
            json.dump(evaluations_master, f, indent=4)
            print(f"Transcript written to {filename}\n")


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