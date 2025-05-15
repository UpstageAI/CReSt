from typing import List, Optional, Tuple

from .baseline import BaselineMethod


class TreeOfThoughtMethod(BaselineMethod):
    """
    RAG QA Tree-of-Thoughts

    Args:
        model_name (str): The name of the OpenAI model to use.
        seed (int): The random seed.
        max_depth (int): The depth of ToT search.
        breadth (int): The number of candidates to retain at each step.
        k (int): The number of thought candidates to generate at once.
    """

    propose_prompt = (
        "Given the question, supporting documents, and current thoughts, "
        "propose {k} next intermediate reasoning steps.\n\n"
        "Question:\n{question}\n\n"
        "Documents:\n{docs}\n\n"
        "Current thoughts:\n{thoughts}\n\n"
        "Return each thought on its own line, without numbering."
    )
    evaluate_prompt = (
        "Given the question, supporting documents, current reasoning steps, and a candidate step,\n"
        "classify the candidate as one of [sure / maybe / impossible] for leading to a correct answer.\n\n"
        "Question:\n{question}\n\n"
        "Documents:\n{docs}\n\n"
        "Current thoughts:\n{thoughts}\n\n"
        "Candidate thought:\n{candidate}\n\n"
        "Just return exactly one word from [sure, maybe, impossible]."
    )
    final_prompt = (
        "You are a helpful assistant tasked with answering questions strictly based on the content of the provided documents. The documents may contain irrelevant or inaccurate information, so please reason carefully and critically when forming your answer.\n"
        "\n"
        "<Rules>\n"
        "1. Only use information that is explicitly stated in the documents. Do not rely on prior knowledge or make assumptions beyond the content.\n"
        "2. If the question cannot be answered solely based on the provided documents, respond with:  \n"
        '    "I cannot answer because the question is unanswerable with the documents."  \n'
        "    Then briefly explain why the information is insufficient.\n"
        "3. Always cite the document numbers used to derive your answer, using the format [1], [2], etc.\n"
        "4. If multiple documents were referenced, include all relevant numbers at the end of your answer.\n"
        "5. Answer the question directly. Do not return any preamble, explanation, or reasoning.\n"
        "</Rules>\n\n"
        "Based on the question, supporting documents, and the chosen reasoning steps, "
        "provide the immediate final concise answer without additional thoughts.\n\n"
        "Documents:\n{docs}\n\n"
        "Question:\n{question}\n\n"
        "Chosen reasoning steps:\n{thoughts}\n\n"
        "Answer:"
    )

    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        max_depth: int = 3,
        breadth: int = 5,
        k: int = 5,
    ):
        self.model_name = model_name
        self.seed = seed
        self.max_depth = max_depth
        self.breadth = breadth
        self.k = k

    def predict(
        self, question: str, docs: List[str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str], int, int]:
        """Predict the answer to the question based on the documents.

        Args:
            question (str): The question to answer.
            docs (List[str]): The documents to use for the answer.

        Returns:
            Tuple[Optional[str], Optional[str], Optional[str], int, int]: The parsed answer, the prompt, the predicted answer, the prompt tokens, and the completion tokens.
        """
        # Initialize token usage counters
        total_prompt_tokens = 0
        total_completion_tokens = 0

        best_path, search_usage = self._search_tree(question, docs)
        total_prompt_tokens += search_usage[0]
        total_completion_tokens += search_usage[1]

        final_answer, prompt, final_usage = self._final_answer(
            question, docs, best_path
        )
        total_prompt_tokens += final_usage[0]
        total_completion_tokens += final_usage[1]

        return (
            final_answer,
            prompt,
            final_answer,  # Using the same final_answer as both parsed and predicted
            total_prompt_tokens,
            total_completion_tokens,
        )

    def _search_tree(
        self, question: str, docs: List[str]
    ) -> Tuple[List[str], Tuple[int, int]]:
        """
        ToT-BFS: docs 포함, 각 단계에서 후보 생성→평가→상위 breadth 선택.

        Returns:
            Tuple[List[str], Tuple[int, int]]: Best path and token usage (prompt_tokens, completion_tokens)
        """
        frontier: List[Tuple[List[str], float]] = [([], 0.0)]
        docs_str = self.serialize_docs(docs)

        # Track token usage
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for _ in range(self.max_depth):
            all_candidates: List[Tuple[List[str], float]] = []
            for thoughts, _ in frontier:
                next_ths, gen_usage = self._generate_thoughts(
                    question, docs_str, thoughts
                )
                total_prompt_tokens += gen_usage[0]
                total_completion_tokens += gen_usage[1]

                for cand in next_ths:
                    score, eval_usage = self._evaluate_thought(
                        question, docs_str, thoughts, cand
                    )
                    total_prompt_tokens += eval_usage[0]
                    total_completion_tokens += eval_usage[1]
                    all_candidates.append((thoughts + [cand], score))

            all_candidates.sort(key=lambda x: x[1], reverse=True)
            frontier = all_candidates[: self.breadth]

        best_path, _ = max(frontier, key=lambda x: x[1]) if frontier else ([], 0.0)
        return best_path, (total_prompt_tokens, total_completion_tokens)

    def _generate_thoughts(
        self, question: str, docs_str: str, thoughts: List[str]
    ) -> Tuple[List[str], Tuple[int, int]]:
        prompt = self.propose_prompt.format(
            question=question,
            docs=docs_str,
            thoughts="\n".join(thoughts) if thoughts else "(none)",
            k=self.k,
        )
        resp, usage, err = self.get_response([{"role": "user", "content": prompt}])
        if err:
            return [], (
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )

        thoughts = [line.strip() for line in resp.splitlines() if line.strip()][
            : self.k
        ]
        return thoughts, (usage.prompt_tokens, usage.completion_tokens)

    def _evaluate_thought(
        self, question: str, docs_str: str, thoughts: List[str], candidate: str
    ) -> Tuple[float, Tuple[int, int]]:
        prompt = self.evaluate_prompt.format(
            question=question,
            docs=docs_str,
            thoughts="\n".join(thoughts) if thoughts else "(none)",
            candidate=candidate,
        )
        resp, usage, err = self.get_response([{"role": "user", "content": prompt}])

        if err:
            return 0.0, (
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )

        label = resp.strip().lower()
        score_map = {"sure": 1.0, "maybe": 0.5, "impossible": 0.0}
        score = score_map.get(label, 0.5)  # Default to maybe if unexpected response

        return score, (usage.prompt_tokens, usage.completion_tokens)

    def _final_answer(
        self, question: str, docs: List[str], thoughts: List[str]
    ) -> Tuple[str, str, Tuple[int, int]]:
        docs_str = self.serialize_docs(docs)
        prompt = self.final_prompt.format(
            question=question,
            docs=docs_str,
            thoughts="\n".join(thoughts),
        )
        resp, usage, err = self.get_response([{"role": "user", "content": prompt}])
        if err:
            return "", (
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )

        return resp.strip(), prompt, (usage.prompt_tokens, usage.completion_tokens)
