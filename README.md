# My Adventures with Large Language Models

**Build foundational LLMs from Transformers to DeepSeek, from scratch, in PyTorch.**

<p align="center">
  <img src="cover.png" alt="My Adventures with Large Language Models" width="400">
</p>

<p align="center">
  <a href="https://leanpub.com/adventures-with-llms">📖 Get the Book</a> ·
  <a href="https://leanpub.com/adventures-with-llms">📄 Free Sample (Ch1)</a> ·
  <a href="https://x.com/S1LV3RJ1NX">𝕏 Follow for Updates</a>
</p>

---

This is the companion code repository for the book [_My Adventures with Large Language Models_](https://leanpub.com/adventures-with-llms) by Prathamesh Saraf.

The book walks you through building five LLM architectures from scratch in PyTorch. Every chapter has runnable, end-to-end code. No pseudocode, no hand-waving. You type it, you run it, you understand it.

## What You'll Build

| Chapter                     | Architecture                        | Key Concepts                                                                                                                                                                                        | Pretrained Weights   |
| --------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| [**Ch 1**](./ch1/README.md) | Vanilla Encoder-Decoder Transformer | Embeddings, sinusoidal PE, scaled dot-product attention, multi-head attention, layer norm, residual connections, English→Hindi translation                                                          | Trained from scratch |
| [**Ch 2**](./ch2/README.md) | GPT-2 (124M)                        | Decoder-only transformer, causal attention, BPE tokenization, greedy and sampling-based decoding, pretraining                                                                                       | ✅ OpenAI weights    |
| [**Ch 3**](./ch3/README.md) | Llama 3.2-3B                        | RMSNorm, RoPE, SwiGLU, Grouped-Query Attention (4 swaps from GPT-2)                                                                                                                                 | ✅ Meta weights      |
| [**Ch 4**](./ch4/README.md) | KV Cache + MQA + GQA                | Inference optimisation, KV cache mechanics, memory-expressiveness tradeoff                                                                                                                          | — (bridge chapter)   |
| [**Ch 5**](./ch5/README.md) | DeepSeek                            | Multi-Head Latent Attention, absorption trick, decoupled RoPE, DeepSeekMoE, shared experts, fine-grained segmentation, auxiliary-loss-free load balancing, Multi-Token Prediction, FP8 quantisation | Trained from scratch |

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

2. Create the uv environment:

```bash
uv sync
```

3. Activate the environment:

```bash
source .venv/bin/activate
```

4. Set up the environment variables. You will need a Hugging Face token to download the models:

```bash
cp .env.sample .env
```

Each chapter has its own README with instructions. For example, to run Chapter 1, go to the `ch1` folder and follow the instructions there.

## Hardware Requirements

- **Chapters 1-2:** Laptop CPU is sufficient. A single GPU speeds up Ch2 pretraining but is not required.
- **Chapter 3:** Loading Llama 3.2-3B needs ~8 GB VRAM in half precision (RTX 3090/4090 or equivalent).
- **Chapters 4-5:** Architectural discussions with small-scale code. Runs on any modern machine.

Every chapter has a small-scale variant that runs on CPU and a full-scale variant for GPU.

## Who This Book Is For

ML engineers, researchers, and senior developers who know Python and PyTorch and want to understand modern LLMs at the level of code. If you've read Raschka or watched Karpathy and want to go further, into Llama, GQA, MLA, and MoE, this is the book.

## The Book

The code here is fully open source. The book provides the explanations, intuitions, derivations, diagrams, and step-by-step narrative that make the code make sense.

**[Get the book on Leanpub →](https://leanpub.com/adventures-with-llms)**

A free sample covering the Preface, full Table of Contents, and Chapter 1 through Causal Attention is available on the book page.

## Errata

Found an error? [Open an issue](https://github.com/S1LV3RJ1NX/mal-code/issues) on this repo.

## Citation

If you use this code in your work:

```bibtex
@book{saraf2026adventures,
  title     = {My Adventures with Large Language Models},
  subtitle  = {Build foundational LLMs from Transformers to DeepSeek, from scratch, in PyTorch},
  author    = {Prathamesh Saraf},
  year      = {2026},
  publisher = {Self-published via Leanpub},
  url       = {https://leanpub.com/adventures-with-llms}
}
```

## Acknowledgements

This book and code build on the work of [Andrej Karpathy](https://x.com/karpathy), [Sebastian Raschka](https://x.com/rasbt), [Umar Jamil](https://x.com/hkproj), and [Raj Dandekar](https://x.com/raj_dandekar). Engineers at [TrueFoundry](https://truefoundry.com/) reviewed drafts and provided guidance throughout.

## License

The code in this repository is released under the MIT License. The book text and illustrations are copyrighted.
