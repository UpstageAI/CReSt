from typing import Any, Dict, List, Literal, Optional, TypedDict, Union


class EvaluationAPIResponse(TypedDict):
    justification: str
    score: int


ERROR_TYPES = [
    "NumericMistakes",
    "MissingDetail",
    "AnswerRefusal",
    "Others",
]


class DocumentInfo(TypedDict):
    # Document ID corresponding to the id of `Document`
    docid: int
    # Document content corresponding to the content of `Document`
    content: str
    # Document format corresponding to the format of `Document`
    type: Literal["text", "html"]
    # Document source path corresponding to the source_path of `Document`
    source_path: str
    # Whether the document is the grounding document for QA or not
    ground: bool
    # Answer citation information
    citation_idx: Optional[List[Dict[str, Any]]]


class QAFinalDatum(TypedDict):
    """A final question-answer pair having retrieved additional docs."""

    # The unique id of the question-answer pair, start from 1
    id: Optional[Union[str, int]]
    # question
    query: str
    # options
    options: Optional[List[str]]
    # answer
    answer: str
    # Explanation for the answer
    explanation: Optional[str]
    # The reasoning type of the question
    reasoning_type: List[str]
    # The type of the question
    question_type: str
    # The selected `Document` chunk ids
    selected_chunk_ids: List[int]
    # The list of `Document` chunks
    documents: List[DocumentInfo]
    # Meta information
    meta: Dict[str, Any]


class PredictionDetails(TypedDict):
    # The model name
    model_name: str
    # The tokenizer name
    max_length: int
    # The number of outputs to generate
    n: int
    # The temperature to use when sampling
    temperature: float
    # Whether to convert HTML to text before generating
    textualize: bool


class QAExampleAnswered(QAFinalDatum):
    """A final question-answer pair having retrieved additional docs and answered."""

    # The predicted answer
    predicted_answer: str
    # The detail options for the prediction
    prediction_details: PredictionDetails


class EvaluationResultMeta(TypedDict):
    model: str
    seed: int
    error: Optional[str]


class EvaluationResult(TypedDict):
    meta: EvaluationResultMeta
    score: float
    error_types: List[str]
    justification: str
    usage: Dict


class QAExampleEvaluated(QAExampleAnswered):
    """A final question-answer pair having retrieved additional docs, answered and evaluated."""

    # The evaluation result
    evaluation_result: EvaluationResult
