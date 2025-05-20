import openai
import anthropic
from google import genai
from google.genai import types

def get_Claude_response(messages, model="claude-3-haiku-20240307", temperature=1.0, max_tokens=1024, return_full_response=False):
    try:
        client = anthropic.Anthropic(api_key="YOUR_API_KEY")

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
        return f"Error in Claude API call: {str(e)}"

def get_OAI_response(messages, model="gpt-4.1-nano-2025-04-14", temperature=1.0, max_tokens=1024, return_full_response=False):
    try:
        client = openai.OpenAI(api_key="YOUR_API_KEY")

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
        return f"Error in OpenAI API call: {str(e)}"
    
def get_Gemini_response(messages, model="gemini-2.0-flash", temperature=1.0, max_tokens=1024, return_full_response=False):

    try:
        client = genai.Client(api_key="YOUR_API_KEY")

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
        return f"Error in Gemini API call: {str(e)}"
    
def get_DeepSeek_response(messages, model="deepseek-chat", temperature=1.0, max_tokens=1024, return_full_response=False):
    try:
        client = openai.OpenAI(api_key="YOUR_API_KEY")

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
        return f"Error in DeepSeek API call: {str(e)}"
        
def get_Grok_response(messages, model="grok-3-mini-beta", temperature=1.0, max_tokens=1024, return_full_response=False):
    try:
        client = openai.OpenAI(api_key="YOUR_API_KEY")

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
        return f"Error in Grok API call: {str(e)}"