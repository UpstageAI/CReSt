from ..prompts.crest import SEMI_STRUCTURED_ANSWER_GENERATION_PROMPT
from .baseline import BaselineMethod


class SemiStructuredMethod(BaselineMethod):
    """Semi-structured method for answer generation.

    Args:
        model_name (str): The name of the model to use.
        seed (int): The seed to use for the answer generation.
        reasoning (bool): Whether to use reasoning model.
    Attributes:
        base_prompt (str): The base prompt to use for the answer generation.
        model_name (str): The name of the model to use.
        seed (int): The seed to use for the answer generation.
        client (OpenAI): The OpenAI client to use.
    """

    base_prompt = SEMI_STRUCTURED_ANSWER_GENERATION_PROMPT
