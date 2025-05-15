KIE_GENERATION_PROMPT = """You will be provided with {chunkcount} chunk(s) of document, each in {format} format.
Your mission is to come up with a series of questions with varying levels of difficulty to test students' understanding of the information contained within these document chunks.
This process will be conducted over multiple steps.

Here are the document chunks:
{content}
[First task]
Your first task is to generate a collection of Key-Value pairs for a Key-Information Extraction (KIE) task.
The extracted key information should have sufficient coverage and be comprehensive enough to reconstruct a text passage that retains the essential ideas and meaning of the original text.
Please provide the Key-Value pairs in the following format:
```json
{{
    "KIE": [
        {{
            "Key": "<Name of key 1>",
            "Value": "[List of values corresponding to the key found in the text chunks]",
            "Description": "<Description of the key to support understanding of the Key-Value pair>"
        }},
        {{
            "Key": "<Name of key 2>",
            "Value": "[List of values corresponding to the key found in the text chunks]",
            "Description": "<Description of the key to support understanding of the Key-Value pair>"
        }},
        ...
    ]    
}}
```

You are to only respond in JSON format and providing the Key-Value pairs for all the chunks supplied.
"""
