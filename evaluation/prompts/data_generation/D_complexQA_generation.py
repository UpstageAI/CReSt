COMPLEXQA_GENERATION_PROMPT = """[Final task]
Now, we will create complex application QAs based on the existing simple application QAs.
These questions differ from simple application QAs by requiring multiple reasoning steps and incorporating a combination of reasoning types to arrive at the answer.

When forming complex/multiple application QAs, follow these guidelines:
	1.	Merge and Modify Thoughtfully:
        •	Combine information from different simple application QAs to form new, complex questions.
        •	Avoid creating trivial questions that are merely a concatenation of existing QAs. Ensure the merged question requires deeper reasoning and processing.
	2.	Step-by-Step Reasoning:
        •	Frame the question so that the student must:
            •	Utilize the reasoning result of one step as an input for the next step.
            •	Apply additional reasoning (numerical, temporal, tabular, multi-constraint, or format reasoning) to arrive at the final answer.
	3.	Challenge and Engagement:
        •	Ensure the QAs challenge the student by requiring them to integrate knowledge and think critically.
        •	Design the reasoning flow to be logical and non-trivial.

The generated complex application QAs should be in the following JSON format:
```json
{
    "complexQA": [
        {
            "id": "<ID of the complexQA, e.g. complexQA1, complexQA2, ...>",
            "question": "<Question text>",
            "answer": "<Answer text>",
            "reasoning_type": "[List of reasoning types that the question tests, e.g. Numerical reasoning, Tabular reasoning, Multi-constraint reasoning, Temporal reasoning, Format reasoning]"
        },
        ...
    ]
}
```

You are to respond in JSON format only and ensure the questions are clear, concise, and require multiple reasoning steps to arrive at the answer.
Generating these QAs could be challenging, but it will help students develop a deeper understanding of the content.
"""