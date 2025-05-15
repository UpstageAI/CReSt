DIRECT_ANSWER_GENERATION_PROMPT = """You are a helpful assistant tasked with answering questions strictly based on the content of the provided documents. The documents may contain irrelevant or inaccurate information, so please reason carefully and critically when forming your answer.

<Rules>
1. Only use information that is explicitly stated in the documents. Do not rely on prior knowledge or make assumptions beyond the content.
2. If the question cannot be answered solely based on the provided documents, respond with:  
   "I cannot answer because the question is unanswerable with the documents."  
   Then briefly explain why the information is insufficient.
3. Always cite the document numbers used to derive your answer, using the format [1], [2], etc.
4. If multiple documents were referenced, include all relevant numbers at the end of your answer.
5. Answer the question directly. Do not return any preamble, explanation, or reasoning.
</Rules>

<Question>  
{question}  
</Question>  

<Documents>  
{docs}  
</Documents>
"""

NAIVE_COT_FOR_REASONING_MODEL_ANSWER_GENERATION_PROMPT = """You are a helpful assistant tasked with answering questions strictly based on the content of the provided documents. The documents may contain irrelevant or inaccurate information, so please reason carefully and critically when forming your answer.

<Rules>
1. Only use information that is explicitly stated in the documents. Do not rely on prior knowledge or make assumptions beyond the content.
2. If the question cannot be answered solely based on the provided documents, respond with:  
   "I cannot answer because the question is unanswerable with the documents."  
   Then briefly explain why the information is insufficient.
3. Always cite the document numbers used to derive your answer, using the format [1], [2], etc.
4. If multiple documents were referenced, include all relevant numbers at the end of your answer.
5. Think step by step to answer the following question.
</Rules>

<Question>  
{question}  
</Question>  

<Documents>  
{docs}  
</Documents>
"""

NAIVE_COT_ANSWER_GENERATION_PROMPT = """You are a helpful assistant tasked with answering questions strictly based on the content of the provided documents. The documents may contain irrelevant or inaccurate information, so please reason carefully and critically when forming your answer.

<Rules>
1. Only use information that is explicitly stated in the documents. Do not rely on prior knowledge or make assumptions beyond the content.
2. If the question cannot be answered solely based on the provided documents, respond with:  
   "I cannot answer because the question is unanswerable with the documents."  
   Then briefly explain why the information is insufficient.
3. Always cite the document numbers used to derive your answer, using the format [1], [2], etc.
4. If multiple documents were referenced, include all relevant numbers at the end of your answer.
5. Think step by step to answer the following question. Return thinking steps between <Thinking> and </Thinking> and the answer between <Answer> and </Answer>.
</Rules>

<Question>  
{question}  
</Question>  

<Documents>  
{docs}  
</Documents>
"""

PLAN_AND_SOLVE_ANSWER_GENERATION_PROMPT = """You are a helpful assistant tasked with answering questions strictly based on the content of the provided documents. The documents may contain irrelevant or inaccurate information, so please reason carefully and critically when forming your answer.

<Rules>
1. Only use information that is explicitly stated in the documents. Do not rely on prior knowledge or make assumptions beyond the content.
2. If the question cannot be answered solely based on the provided documents, respond with:  
   "I cannot answer because the question is unanswerable with the documents."  
   Then briefly explain why the information is insufficient.
3. Always cite the document numbers used to derive your answer, using the format [1], [2], etc.
4. If multiple documents were referenced, include all relevant numbers at the end of your answer.
5. First understand the problem and devise a plan to solve the problem. Then, carry out the plan to solve the problem step by step. Return intermidiate steps between <Thinking> and </Thinking> and the answer between <Answer> and </Answer>.
</Rules>

<Question>  
{question}  
</Question>  

<Documents>  
{docs}  
</Documents>
"""


COD_ANSWER_GENERATION_PROMPT = """You are a helpful assistant tasked with answering questions strictly based on the content of the provided documents. The documents may contain irrelevant or inaccurate information, so please reason carefully and critically when forming your answer.

<Rules>
1. Only use information that is explicitly stated in the documents. Do not rely on prior knowledge or make assumptions beyond the content.
2. If the question cannot be answered solely based on the provided documents, respond with:  
   "I cannot answer because the question is unanswerable with the documents."  
   Then briefly explain why the information is insufficient.
3. Always cite the document numbers used to derive your answer, using the format [1], [2], etc.
4. If multiple documents were referenced, include all relevant numbers at the end of your answer.
5. Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return thinking steps between <Thinking> and </Thinking> and the answer between <Answer> and </Answer>.
</Rules>

<Question>  
{question}  
</Question>  

<Documents>  
{docs}  
</Documents>
"""

SEMI_STRUCTURED_ANSWER_GENERATION_PROMPT = """Please answer the following question based on the given documents.
The given documents might include noise or inaccurate content. Your task is to answer the question by performing complex reasoning based on the given documents.
If the question cannot be answered based on the given documents, generate exactly ‘I can not answer because the question is "unanswerable" with the documents.’ in English.
Think step by step to answer the following question. 
First, Search for information in the text that is relevant to the question. Then, extract the information that is relevant to the question.
Second, Think if there is any prior / common knowledge that can help us answer the question.
Finally, Think carefully and reason through step-by-step based on what we know so far to answer the question.
Return thinking steps between <Thinking> and </Thinking> and the answer between <Answer> and </Answer>.
<Question>
{question}
</Question>
<Documents>
{docs}
</Documents>
"""

NON_REFUSAL_EVALUATION_PROMPT = """You are an evaluator for a Retrieval Question Answering (QA) task. Your task is to assess how closely the predicted answer matches the golden answer.

**Evaluation Categories:**
- **Correct**: The predicted answer is a perfect match or semantically identical to the golden answer.
- **Partially Correct**: The predicted answer contains some key information from the golden answer but may be incomplete, missing details, or only partially aligned.
- **Wrong**: The predicted answer is completely incorrect, missing essential details, or contains misleading information.

**Consider the following factors when evaluating:**
- **Exactness**: Does the predicted answer exactly match the golden answer?
- **Paraphrasing**: If reworded, does it retain the same meaning?
- **Completeness**: Is the full answer provided, or is it partial?
- **Incorrect Information**: Does the predicted answer introduce any false or misleading details?

**Error Category Guidelines:**
*If the evaluation is not **Correct** (i.e., it is either "Partially Correct" or "Wrong"), also identify the most severe error type present by providing an **ErrorType** field. This field should contain one of the following categories that best describes the main error:*

- **AnswerRefusal**: The answer refuses to provide a response or gives up on answering, despite a clear expectation to do so.
- **NumericMistakes**: The answer contains incorrect arithmetic or inaccurate numeric references (e.g., population sizes, years, differences in ages, championship tallies).
- **MissingDetail**: The answer shows a partial understanding by identifying the correct domain or background but omitting the necessary numeric or textual detail.
- **Others**: Any other error types not covered by the above categories.

**Input:**
- **Question**: {question}
- **Golden Answer**: {golden_answer}
- **Predicted Answer**: {predicted_answer}

**Your response should be formatted as follows:**

```plaintext
**Justification**: <brief explanation of your evaluation>
**Decision**: <Correct/Partially Correct/Wrong>
**ErrorType**: <AnswerRefusal/NumericMistakes/MissingDetail/Others>  // Keep empty if the evaluation is **Correct**
```
"""
