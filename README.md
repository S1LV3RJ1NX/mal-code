# My Adventures with LLM

This repository contains the code for all the chapters of `My adventures with LLM` book.

## Setup

1. Install uv: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

1. Create the uv environment:

```bash
uv sync
```

1. Activate the environment:

```bash
source .venv/bin/activate
```

1. Setup the environment variables. You will need a Hugging Face token to download the models.

```bash
cp .env.sample .env
```

## Run the code

Specific chapters have their own README with instructions on how to run the code.

For example, to run the code for chapter 1, go to the `ch1` folder and follow the instructions in the README.

## Chapters

- [Chapter 1: A Journey into the Heart of Transformers](./ch1/README.md)
- [Chapter 2: LLMs are Unsupervised Multitask Learners](./ch2/README.md)
- [Chapter 3: The Llama3 Herd of Models](./ch3/README.md)
