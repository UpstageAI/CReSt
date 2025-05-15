import json
import os
import re
import shutil
import string
import sys
from typing import Iterable, Optional, Tuple

from datasets import Dataset
from openai import NOT_GIVEN, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import TypeAdapter
from dotenv import load_dotenv

load_dotenv()


def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text):
    return " ".join(text.split())


def remove_citation_from_answer(text):
    return re.sub(r"(?:\s*\[\d+\],?)+\s*$", "", text)


def handle_punc(text):
    exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
    return "".join(ch if ch not in exclude else " " for ch in text)


def lower(text):
    return text.lower()


def replace_underscore(text):
    return text.replace("_", " ")


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    return white_space_fix(
        remove_articles(handle_punc(lower(replace_underscore(s))))
    ).strip()


def extract_json_data(
    output_text: str, validate_cls: Optional[type] = None
) -> Optional[dict]:
    """Parse the JSON data from the output text

    Args:
        output_text (str): The output text from the OpenAI API
    Returns:
        Optional[Dict]: The parsed JSON form QA data
    """
    if "{" not in output_text or "}" not in output_text:
        return None

    text = "{" + output_text.split("{", 1)[1].split("}", 1)[0] + "}"

    try:
        parsed = json.loads(text)
        if validate_cls is not None:
            return TypeAdapter(validate_cls).validate_python(parsed)
        return parsed
    except Exception:
        return None


def openai_chat_completion(
    model_name: str,
    messages: Iterable[ChatCompletionMessageParam],
    seed: Optional[int] = NOT_GIVEN,
    json_response: bool = False,
) -> Tuple[str, dict]:
    """OpenAI Chat Completion API Call

    Args:
        model_name (str): The model name
        messages (Iterable[ChatCompletionMessageParam]): The messages
        seed (Optional[int], optional): The seed. Defaults to NOT_GIVEN.
        json_response (bool, optional): The JSON response. Defaults to False.
    Returns:
        Tuple[str, dict]: The output and the usage
    """
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            seed=seed,
            response_format={"type": "json_object"} if json_response else NOT_GIVEN,
        )
        output = response.choices[0].message.content
        usage = response.usage
        return output, usage, None
    except Exception as e:
        print(f"[!] API Call Error: {e}", file=sys.stderr)
        return None, None, e


def safe_save(dataset: Dataset, output_path: str) -> None:
    """Safely save the dataset to the output path

    Args:
        dataset (Dataset): The dataset to save
        output_path (str): The output path
    """
    if not os.path.exists(output_path):
        dataset.save_to_disk(output_path)
    else:
        tmp_path = output_path.rstrip("/") + ".tmp"
        dataset.save_to_disk(tmp_path)
        shutil.rmtree(output_path)
        os.rename(tmp_path, output_path)
