SIMPLEAPPQA_GENERATION_PROMPT = """[Third task]
Your next task is to refer to the simple Q&A pairs created in the previous turn and generate a new set of Q&A pairs that require application reasoning.
Application reasoning refers to Q&As that require additional steps or though processes to answer, as opposed to simple query-and-fetch Q&A.
Each question should be clear, consider and aligned with the characteristics and reasoning types described below:

Reasoning Types:
1. Numerical reasoning: The question requires the reader to perform arithmetic operations on the information provided in the document, such as counting, comparisons, calculations, etc.
2. Tabular reasoning: The question requires the reader to compare and contrast information across different tables, rows, columns, etc.
3. Multi-constraint reasoning: The question which contains multiple conditions / constraints which require readers to find the answer that satisfies all the conditions / constraints.
4. Temporal reasoning: The question requires the reader to reason about the time-based information provided in the document.
5. Format reasoning: The question requires the reader to reason about the format / post-processing of the information provided in the document (e.g. conversion of units, etc.).

Modify existing or refer to the simple Q&A pairs, or add new ones to incorporate application reasoning types.
Ensure each Q&A is self-contained and does not explicitly reference the document (e.g., avoid phrases like “In the document…”).
You are to response in the following JSON format:
```json
{
    "simpleAppQA": [
        {
            "id": "<ID of the simpleAppQA, e.g. simpleAppQA1, simpleAppQA2, ...>",
            "question": "<Question text>",
            "answer": "<Answer text>",
            "reasoning_type": "<Reasoning type of the question, e.g. Numerical reasoning, Tabular reasoning, Multi-constraint reasoning, Temporal reasoning, Format reasoning>"
        },
        ...
    ]
}
```

You are to respond in JSON format only and ensure the questions are clear, concise, and require reasoning beyond simple query-and-fetch.
"""