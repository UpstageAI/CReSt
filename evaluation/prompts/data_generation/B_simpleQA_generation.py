SIMPLEQA_GENERATION_PROMPT = """[Second task]
Your task is to create a set of simple Q&A pairs using the key-value pairs extracted from the document, while also referring to the original document chunks.
These Q&A pairs should align with the following characteristics:
    1.	Question Construction: Each question should focus on the key from the key-value pairs, while the corresponding value provides the answer.
	2.	Contextual Independence: The questions should not explicitly reference the document or assume its availability. This is because the students answering these questions are expected to have internalized the document’s content. Phrases like “In the document…” should be avoided.
    3.	Reasoning Type: Each question should test one of the following reasoning types:
	    •	Form/Layout Understanding: The question assesses the student’s ability to comprehend the document’s layout or structure, rather than just its text content.
	    •   Tabular Understanding: The question evaluates the student’s ability to interpret and extract information from tabular data.
	    •	Text/Semantic Understanding: The question examines the student’s grasp of the textual content’s meaning and implications.

The Q&A pairs should be in the following format:
```json
{{
    "simpleQA": [
        {{
            "id": "<ID of the simpleQA, e.g. simpleQA1, simpleQA2, ...>",
            "question": "<Question text>",
            "answer": "<Answer text>",
            "reasoning_type": "[List of reasoning types that the question tests, e.g. Form/Layout Understanding, Tabular Understanding, Text/Semantic Understanding]"
        }},
        ...
    ]
}}
```

You are to respond in JSON format only and ensure the questions are clear, concise, and directly tied to the key-value pairs provided.
"""