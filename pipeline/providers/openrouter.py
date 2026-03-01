"""OpenRouter client helper."""

from __future__ import annotations

import os

import openai
from dotenv import load_dotenv


def get_openrouter_response(
    messages,
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    return_full_response: bool = False,
):
    """Send one chat completion request via OpenRouter."""

    try:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        if return_full_response:
            return response
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenRouter API call: {str(e)}")
        return f"Error in OpenRouter API call: {str(e)}"
