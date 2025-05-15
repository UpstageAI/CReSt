import argparse
import json
from datetime import datetime
from functools import partial

from datasets import Features, Value, load_dataset
from evaluation.crest import (
    aggregate_score,
    aggregate_citation_score,
    non_refusal_evaluation,
    predict,
    refusal_evaluation,
    citation_evaluation,
    calculate_unified_score,
)
from evaluation.method import (
    BaselineMethod,
    CoTMethod,
    CoDMethod,
    SemiStructuredMethod,
    TreeOfThoughtMethod,
    LeastToMostMethod,
    PlanAndSolveMethod,
)
from evaluation.utils import safe_save

parser = argparse.ArgumentParser("Evaluation of CReSt")
parser.add_argument(
    "--dataset",
    type=str,
    default="upstage/CReSt",
    help="Dataset path",
)
parser.add_argument(
    "--model",
    type=str,
    default="gpt-4o-mini",
    help="model name",
)
parser.add_argument(
    "--eval-model",
    type=str,
    default="gpt-4o",
    help="OpenAI model name to use for tagging",
)
parser.add_argument(
    "--method",
    type=str,
    default="direct",
    choices=[
        "direct",
        "cot",
        "cot_reasoning_model",
        "cod",
        "semi_structured",
        "tot",
        "least_to_most",
        "plan_and_solve",
    ],
    help="Answer generation method",
)
parser.add_argument(
    "--num-parallels", type=int, default=16, help="Number of parallel processes to use"
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)
parser.add_argument("--output-path", type=str, help="Path to save the output")
parser.add_argument("--n", type=int, default=1, help="Number of answers to generate")
parser.add_argument(
    "--temperature", type=float, default=0.0, help="Temperature for answer generation"
)
parser.add_argument(
    "--overwrite-evaluate",
    action="store_true",
    help="Overwrite the evaluation results",
)
parser.add_argument(
    "--num-samples", type=int, default=None, help="Number of samples to evaluate"
)
parser.add_argument(
    "--lang",
    type=str,
    default=["en", "ko"],
    choices=["en", "ko"],
    nargs="+",
    help="Language to evaluate",
)


def main(args: argparse.Namespace):
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"[+] Dataset loaded. Use language: {args.lang}")
    dataset = dataset.filter(lambda x: json.loads(x["meta"])["language"] in args.lang)

    if args.method == "direct":
        method = BaselineMethod(args.model, args.seed)
    elif args.method == "cot_reasoning_model":
        method = CoTMethod(args.model, args.seed, reasoning=True)
    elif args.method == "cot":
        method = CoTMethod(args.model, args.seed)
    elif args.method == "cod":
        method = CoDMethod(args.model, args.seed)
    elif args.method == "semi_structured":
        method = SemiStructuredMethod(args.model, args.seed)
    elif args.method == "tot":
        method = TreeOfThoughtMethod(args.model, args.seed)
    elif args.method == "least_to_most":
        method = LeastToMostMethod(args.model, args.seed)
    elif args.method == "plan_and_solve":
        method = PlanAndSolveMethod(args.model, args.seed)
    else:
        raise ValueError(f"Invalid method: {args.method}")

    cached = False
    for split in ("refusal", "non_refusal"):
        if args.num_samples is not None:
            dataset[split] = (
                dataset[split].shuffle(args.seed).select(range(args.num_samples))
            )
        if "predicted_answer" in dataset[split].column_names:
            print(f"[+] {split} split is already predicted. Skipping...")
            cached = True
            continue
        print(f"[+] Generating {len(dataset[split])} Answers for {split} split...")

        dataset[split] = dataset[split].map(
            partial(predict, method=method),
            num_proc=args.num_parallels,
            keep_in_memory=True,
            desc="Predicting answers with OpenAI API",
        )
        # filter
        dataset[split] = dataset[split].filter(
            lambda x: x["predicted_answer"] != "", num_proc=args.num_parallels
        )
        print(f"[+] {split} split has {len(dataset[split])} answers")

    if args.output_path is None:
        if cached:
            print(
                "[+] Save path is not provided. Overwriting the input dataset by default."
            )
            args.output_path = args.dataset
        else:
            model_base_name = args.model.split("/")[-1]
            eval_model = args.eval_model.split("/")[-1]
            timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
            args.output_path = f"outputs/{model_base_name}_{args.method}_{eval_model}_{timestamp}"
            print(f"[+] Save path is not provided. Saving to {args.output_path}")

    if not cached:
        print("[+] Saving model prediction results...")
        safe_save(dataset, args.output_path)

    print("[+] Evaluating Answers...")
    eval_cached = True
    if (
        "evaluation_result" not in dataset["refusal"].column_names
        or args.overwrite_evaluate
    ):
        dataset["refusal"] = dataset["refusal"].map(
            refusal_evaluation,
            num_proc=args.num_parallels,
            keep_in_memory=True,
            desc="Evaluating refusal split",
        )
        eval_cached = False
    else:
        print("[!] Refusal evaluation is already done. Skipping...")

    if (
        "evaluation_result" not in dataset["non_refusal"].column_names
        or args.overwrite_evaluate
    ):
        dataset["non_refusal"] = dataset["non_refusal"].map(
            partial(non_refusal_evaluation, model_name=args.eval_model, seed=args.seed),
            num_proc=args.num_parallels,
            keep_in_memory=True,
            desc="Evaluating non_refusal split",
            features=Features(
                {
                    **dataset["non_refusal"].features,
                    "evaluation_result": {
                        "meta": {
                            "model": Value("string"),
                            "seed": Value("int32"),
                            "error": Value("string"),
                        },
                        "score": Value("float32"),
                        "error_type": Value("string"),
                        "usage": {
                            "prompt_tokens": Value("int32"),
                            "completion_tokens": Value("int32"),
                        },
                        "justification": Value("string"),
                    },
                }
            ),
        )
        eval_cached = False
    else:
        print("[!] non_refusal evaluation is already done. Skipping...")
    # Citation
    dataset["non_refusal"] = dataset["non_refusal"].map(
        citation_evaluation,
        num_proc=args.num_parallels,
        keep_in_memory=True,
        desc="Evaluating citation split",
    )

    if not eval_cached:
        print("[+] Saving Final Result...")
        safe_save(dataset, args.output_path)

    print("[+] Aggregating Metrics...")
    for split in ("refusal", "non_refusal"):
        for language in ("en", "ko"):
            for difficulty_type in ("SimpleQA", "ComplexQA"):
                correct_rate, partially_correct_rate, wrong_rate, num_invalid = (
                    aggregate_score(dataset[split], language, [difficulty_type])
                )
                print(
                    f"[{split}] {language}/{difficulty_type} [Correct/Partial/Wrong]: {correct_rate:.2%}, {partially_correct_rate:.2%}, {wrong_rate:.2%}"
                    f" [Invalid] {num_invalid}"
                )
                if split == "non_refusal":
                    citation_precision, citation_recall = aggregate_citation_score(
                        dataset[split], language, difficulty_type
                    )
                    print(
                        f"[citation] {language}/{difficulty_type} Precision/Recall: {citation_precision:.2%}, {citation_recall:.2%}"
                    )
            correct_rate, partially_correct_rate, wrong_rate, num_invalid = (
                aggregate_score(dataset[split], language)
            )
            print(
                f"[{split}] {language}/all [Correct/Partial/Wrong]: {correct_rate:.2%}, {partially_correct_rate:.2%}, {wrong_rate:.2%}"
                f" [Invalid] {num_invalid}"
            )

    # Calculate Unified Score
    for language in ("en", "ko"):
        for difficulty_type in ("SimpleQA", "ComplexQA"):
            unified_score = calculate_unified_score(dataset, language, [difficulty_type])
            print(f"[{language}/{difficulty_type}] Unified Score: {unified_score:.2%}")
        unified_score = calculate_unified_score(
            dataset, language, ["SimpleQA", "ComplexQA"]
        )
        print(f"[{language}/all] Unified Score: {unified_score:.2%}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
