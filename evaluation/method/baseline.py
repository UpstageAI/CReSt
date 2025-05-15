import sys
from typing import Dict, List, Optional

from ..prompts.crest import DIRECT_ANSWER_GENERATION_PROMPT
from ..utils import openai_chat_completion


class BaselineMethod:
    """Naive method for answer generation.

    Args:
        model_name (str): The name of the model to use.
        seed (int): The seed to use for the answer generation.
    Attributes:
        base_prompt (str): The base prompt to use for the answer generation.
        model_name (str): The name of the model to use.
        seed (int): The seed to use for the answer generation.
    """

    base_prompt = DIRECT_ANSWER_GENERATION_PROMPT

    def __init__(self, model_name: str, seed: int = 42):
        self.model_name = model_name
        self.seed = seed

    def predict(
        self, question: str, docs: List[str]
    ) -> tuple[Optional[str], Optional[dict], Optional[Exception]]:
        """Predict the answer to the question based on the documents.

        Args:
            question (str): The question to answer.
            docs (List[str]): The documents to use for the answer.

        Returns:
            tuple[str, str, str, int, int]: The parsed answer, the prompt, the predicted answer, the prompt tokens, and the completion tokens.
        """
        prompt = self.base_prompt.format(
            question=question, docs=self.serialize_docs(docs)
        )
        messages = [{"role": "user", "content": prompt}]
        predicted_answer, usage, error = self.get_response(messages)

        if error is not None:
            print(f"[!] Error with question: {question}", file=sys.stderr)
            return "", prompt, "", 0, 0

        parsed_answer = predicted_answer.split("<Answer>")[-1].split("</Answer>")[0]

        return (
            parsed_answer,
            prompt,
            predicted_answer,
            usage.prompt_tokens,
            usage.completion_tokens,
        )

    def get_response(
        self, messages: List[Dict]
    ) -> tuple[Optional[str], Optional[dict], Optional[Exception]]:
        return openai_chat_completion(
            model_name=self.model_name,
            messages=messages,
            seed=self.seed,
        )

    @staticmethod
    def serialize_docs(docs: List[str]) -> str:
        return "\n".join([f"[{i + 1}]\n{doc}" for i, doc in enumerate(docs)])
