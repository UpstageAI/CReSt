from ..prompts.crest import PLAN_AND_SOLVE_ANSWER_GENERATION_PROMPT
from .baseline import BaselineMethod


class PlanAndSolveMethod(BaselineMethod):
    """RAG QA using Plan-and-Solve prompting, extending BaselineMethod."""

    def __init__(self, model_name: str, seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        self.base_prompt = PLAN_AND_SOLVE_ANSWER_GENERATION_PROMPT
