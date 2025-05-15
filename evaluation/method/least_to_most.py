import json
import sys
from typing import Dict, List, Optional, Tuple

from .baseline import BaselineMethod


class LeastToMostMethod(BaselineMethod):
    """RAG QA using least-to-most prompting, extending NaiveMethod with staged methods."""

    decomposition_template: str = (
        "You are an assistant that decomposes a question into simpler sub-questions.\n"
        "Context:\n{docs}\n"
        "Question: {question}\n"
        "Return each sub-question on its own line, without numbering."
    )

    solving_context_template: str = (
        "You are a helpful assistant tasked with answering questions strictly based on the content of the provided documents. The documents may contain irrelevant or inaccurate information, so please reason carefully and critically when forming your answer.\n"
        "Rules:\n"
        "1. Only use information that is explicitly stated in the documents. Do not rely on prior knowledge or make assumptions beyond the content.\n"
        "2. If the question cannot be answered solely based on the provided documents, respond with: \n"
        "   I cannot answer because the question is unanswerable with the documents.\n"
        "   Then briefly explain why the information is insufficient.\n"
        "3. Always cite the document numbers used to derive your answer, using the format [1], [2], etc.\n"
        "4. If multiple documents were referenced, include all relevant numbers at the end of your answer.\n"
        "5. Think step by step to answer the following question.\n"
        "</Rules>\n"
        "<Documents>\n"
        "{docs}\n"
        "</Documents>\n"
    )
    solving_template: str = (
        "Answer the following question using the information provided in the context.\n"
        "<Question>\n"
        "{sub_question}\n"
        "</Question>\n"
    )

    def predict(
        self, question: str, docs: List[str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str], int, int]:
        """Main entry: performs decomposition and solving stages."""
        docs_string = self.serialize_docs(docs)

        # Stage 1: Decompose
        sub_questions, dec_prompt, dec_resp, dec_usage, dec_error = self._decompose(
            question, docs_string
        )
        if dec_error or not sub_questions:
            return (
                None,
                dec_prompt,
                dec_resp,
                dec_usage.prompt_tokens if dec_usage else 0,
                dec_usage.completion_tokens if dec_usage else 0,
            )

        # Stage 2: Solve
        sub_questions.append(question)
        final_answer, total_tokens = self._solve_subquestions(
            sub_questions, docs_string
        )
        total_tokens["prompt_tokens"] += dec_usage.prompt_tokens
        total_tokens["completion_tokens"] += dec_usage.completion_tokens

        parsed_answer = final_answer.split("<Answer>")[-1].split("</Answer>")[0]

        return (
            parsed_answer,
            dec_prompt,
            dec_resp,
            total_tokens["prompt_tokens"],
            total_tokens["completion_tokens"],
        )

    def _decompose(
        self, question: str, docs_string: str
    ) -> Tuple[Optional[List[str]], str, Optional[str], Optional[object], bool]:
        """Stage 1: generate sub-questions from the main question."""
        prompt = self.decomposition_template.format(docs=docs_string, question=question)
        messages = [{"role": "user", "content": prompt}]
        resp, usage, error = self.get_response(messages)
        if error:
            print(f"[!] Decomposition error for question: {question}", file=sys.stderr)
            return None, prompt, None, None, True
        try:
            sub_questions = [q for q in resp.strip().split("\n") if q.strip()]
        except Exception:
            print(f"[!] Failed to parse sub-questions: {resp}", file=sys.stderr)
            return None, prompt, resp, usage, True
        return sub_questions, prompt, resp, usage, False

    def _solve_subquestions(
        self, sub_questions: List[str], docs_string: str
    ) -> Tuple[str, Dict[str, int]]:
        """Stage 2: sequentially solve each sub-question and aggregate tokens."""
        prompt_tokens = 0
        completion_tokens = 0

        messages = [
            {
                "role": "user",
                "content": self.solving_context_template.format(docs=docs_string),
            }
        ]
        for sub_question in sub_questions:
            prompt = self.solving_template.format(sub_question=sub_question)
            messages.append({"role": "user", "content": prompt})

            response, usage, error = self.get_response(messages)
            if error:
                print(
                    f"[!] Solving error for sub-question: {sub_question}",
                    file=sys.stderr,
                )
                break

            messages.append({"role": "assistant", "content": response})
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens

        total_tokens = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        return response, total_tokens
