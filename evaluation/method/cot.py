from ..prompts.crest import (
    NAIVE_COT_ANSWER_GENERATION_PROMPT,
    NAIVE_COT_FOR_REASONING_MODEL_ANSWER_GENERATION_PROMPT,
)
from .baseline import BaselineMethod


class CoTMethod(BaselineMethod):
    """CoT method for answer generation.

    Args:
        model_name (str): The name of the model to use.
        seed (int): The seed to use for the answer generation.
        reasoning (bool): Whether to use reasoning model.
    Attributes:
        base_prompt (str): The base prompt to use for the answer generation.
        model_name (str): The name of the model to use.
        seed (int): The seed to use for the answer generation.
    """

    def __init__(self, model_name: str, seed: int = 42, reasoning: bool = False):
        self.model_name = model_name
        self.seed = seed
        self.base_prompt = (
            NAIVE_COT_ANSWER_GENERATION_PROMPT
            if not reasoning
            else NAIVE_COT_FOR_REASONING_MODEL_ANSWER_GENERATION_PROMPT
        )
