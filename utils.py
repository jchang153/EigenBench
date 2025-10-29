import openai
import anthropic
from google import genai
from google.genai import types
import math
import re

from dotenv import load_dotenv
import os

def get_Claude_response(messages, model="claude-3-haiku-20240307", temperature=1.0, max_tokens=1024, return_full_response=False):
    try:
        load_dotenv()
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        system_messages=[]
        user_messages=[]

        for message in messages:
            role = message['role']
            content = message['content']

            if role=='system':
                system_messages.append(content)
            elif role=='user':
                user_messages.append({"role": role, "content": content})

        if len(system_messages) != 0:
            response = client.messages.create(
                model=model,
                messages=user_messages,
                system="\n".join(system_messages),
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            response = client.messages.create(
                model=model,
                messages=user_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        if return_full_response:
            return response
        else:
            return response.content[0].text

    except Exception as e:
        print(f"Error in Claude API call: {str(e)}")
        return f"Error in Claude API call: {str(e)}"

def get_OAI_response(messages, model="gpt-4.1-nano-2025-04-14", temperature=1.0, max_tokens=1024, return_full_response=False, log_probs=False):
    try:
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        if "o4" in model or "o3" in model or "o1" in model:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                max_completion_tokens=512, # accounts for both reasoning and output tokens; from my testing needs to be at least 512
                reasoning_effort="low"
            )
        else:
            if log_probs:
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    max_tokens=max_tokens,
                    logprobs=True,
                    top_logprobs=5
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    max_tokens=max_tokens
                )
        if return_full_response:
            return response # the full response object
        else:
            return response.choices[0].message.content # just the generated text
    except Exception as e:
        print(f"Error in OpenAI API call: {str(e)}")
        return f"Error in OpenAI API call: {str(e)}"
    
def get_Gemini_response(messages, model="gemini-2.0-flash", temperature=1.0, max_tokens=1024, return_full_response=False):

    try:
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=GEMINI_API_KEY)

        system_messages=[]
        user_messages=[]

        for message in messages:
            role = message['role']
            content = message['content']

            if role=='system':
                system_messages.append(content)
            elif role=='user':
                user_messages.append(content)

        if len(system_messages) != 0:
            response = client.models.generate_content(
                model=model,
                contents=user_messages,
                config=types.GenerateContentConfig(
                    system_instruction=system_messages,
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )
        else:
            response = client.models.generate_content(
                model=model,
                contents=user_messages,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )
        if return_full_response:
            return response # the full response object
        else:
            return response.text # just the generated text

    except Exception as e:
        print(f"Error in Gemini API call: {str(e)}")
        return f"Error in Gemini API call: {str(e)}"
    
def get_DeepSeek_response(messages, model="deepseek-chat", temperature=1.0, max_tokens=1024, return_full_response=False):
    try:
        load_dotenv()
        DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        client = openai.OpenAI(api_key=DEEPSEEK_API_KEY,
                               base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens
        )
        if return_full_response:
            return response # the full response object
        else:
            return response.choices[0].message.content # just the generated text
    except Exception as e:
        print(f"Error in DeepSeek API call: {str(e)}")
        return f"Error in DeepSeek API call: {str(e)}"
        
def get_Grok_response(messages, model="grok-3-mini-beta", temperature=1.0, max_tokens=1024, return_full_response=False):
    try:
        load_dotenv()
        GROK_API_KEY = os.getenv("GROK_API_KEY")
        client = openai.OpenAI(api_key=GROK_API_KEY,
                               base_url="https://api.x.ai/v1")

        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens, # after testing, it seems like we need at least max_tokens=1024 to get an actual response. Or set to None
            # reasoning_effort="low" # comment this out if using grok 4
        )
        if return_full_response:
            return response # the full response object
        else:
            return response.choices[0].message.content # just the generated text
    except Exception as e:
        print(f"Error in Grok API call: {str(e)}")
        return f"Error in Grok API call: {str(e)}"
    

def get_OpenRouter_response(messages, model, temperature=1.0, max_tokens=1024, return_full_response=False):
    try:
        load_dotenv()
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        )

        response = client.chat.completions.create(
            model=model,
            temperature = temperature,
            max_tokens = max_tokens,
            messages=messages
        )
        if return_full_response:
            return response # the full response object
        else:
            return response.choices[0].message.content # just the generated text

    except Exception as e:
        print(f"Error in OpenRouter API call: {str(e)}")
        return f"Error in OpenRouter API call: {str(e)}"
    
def get_choice_token_logprobs(response):
    
    content = response.choices[0].message.content
    logprobs_data = response.choices[0].logprobs.content
    
    choice_match = re.search(r'<choice>(\d)</choice>', content)
    if not choice_match:
        return None
    
    chosen_number = choice_match.group(1)
    choice_start = choice_match.start() + len('<choice>')
    choice_end = choice_match.end() - len('</choice>')
    
    # Reconstruct token positions to find the exact choice token
    current_pos = 0
    target_token_idx = None
    
    for i, token_logprob in enumerate(logprobs_data):
        token = token_logprob.token
        token_end = current_pos + len(token)
        
        # Check if this token overlaps with the choice number position
        if current_pos <= choice_start < token_end and chosen_number in token:
            target_token_idx = i
            break
        current_pos = token_end
    
    if target_token_idx is not None and logprobs_data[target_token_idx].top_logprobs:
        result = {}
        for alt in logprobs_data[target_token_idx].top_logprobs:
            if alt.token in ['1', '2']:
                result[alt.token] = {
                    'logprob': alt.logprob,
                    'prob': math.exp(alt.logprob)
                }
        
        # Calculate normalized probability for choice 1
        if '1' in result and '2' in result:
            prob_1 = result['1']['prob']
            prob_2 = result['2']['prob']
            result['normalized_prob_1'] = prob_1 / (prob_1 + prob_2)
        
        return result
    
    return None