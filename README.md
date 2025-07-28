# CReSt Benchmark
CReSt: A Comprehensive Benchmark for Retrieval-Augmented Generation with Complex Reasoning over Structured Documents

## ⚡️ Introduction
CReSt is a benchmark consisting of 2,245 human-annotated examples in English and Korean, designed to capture complex, multi-step RAG scenarios.

You can explore the dataset on Hugging Face at: [https://huggingface.co/datasets/upstage/CReSt](https://huggingface.co/datasets/upstage/CReSt)

## 📣 Latest Updates

- [15/05/2025] Release of CReSt code

## 🚀 Quick Start
1. Clone the repository and install the required dependencies.

```shell
git clone git@github.com:UpstageAI/CReSt.git
cd CReSt
pip install -r requirements.txt
```

2. Copy the .env.example template and rename it to .env. Then, update it with your API keys.
```shell
cp .env.example .env
``` 

3. Run the script.
```shell
python -m scripts.run_evaluation --model $MODEL \
                                 --eval-model gpt-4o \
                                 --method $METHOD \
                                 --dataset upstage/CReSt
```

## 📜 License
This benchmark is distributed under the CC-by-NC 4.0.

## 📝 Citation
If you use this code in your research, please cite:
```
@inproceedings{khang2025crest,
  title={CReSt: A Comprehensive Benchmark for Retrieval-Augmented Generation with Complex Reasoning over Structured Documents},
  author={Khang, Minsoo and Park, Sangjun and Hong, Teakgyu and Jung, Dawoon},
  booktitle={TBD},
  pages={TBD},
  year={2025}
}
```
