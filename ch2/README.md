# Chapter 2: LLMs are Unsupervised Multitask Learners

_Note:_ Make sure to follow the instructions in the [README](../README.md) to install the dependencies.

## Overview

The goal of this chapter is to pre-train a transformer model on a small corpus of text for text-generation task. Specifically, we will implement the [GPT-2](./02_language_models_are_unsupervised_multitask_learners.pdf) model.

## Folder Structure

- `ch2/`
  - `components/` - Components of the transformer model (encoder, decoder, attention, etc.)
  - `load_pretrained.py` - Load the pre-trained model
  - `dataset.py` - Dataset class for loading and preprocessing the data
  - `trainer.py` - Training script
  - `inference.py` - Inference script
  - `utils.py` - Utility functions

## Training

You can either train the model from scratch or use pre-trained weights from [Hugging Face](https://huggingface.co/s1lv3rj1nx/ch2). To train the model from scratch, run the following command:

```bash
python trainer.py
```

If you want to use pre-trained weights to generate text, download the `best_model.pth` from the [Hugging Face](https://huggingface.co/s1lv3rj1nx/ch2) and place it in the `ch2/checkpoints` directory

Since, we have pre-trained on a small amount of data, the model has overfitted, but can still generate sensible text.

## Plots

Loss (Train):

![ch2_05_train_epoch_loss.png](https://cdn-uploads.huggingface.co/production/uploads/62790519541f3d2dfa79a6cb/Ht1Tfjuoqywbf5GF06jMx.png)

Perplexity (Train):
![image/png](https://cdn-uploads.huggingface.co/production/uploads/62790519541f3d2dfa79a6cb/psCddxI08z64FKzPH3ADk.png)

Loss (Val):

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62790519541f3d2dfa79a6cb/Ul5sRV2g0HT2CTCU1FQBT.png)

Perplexixty (Val):

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62790519541f3d2dfa79a6cb/TmZ6cn7g48q3sAjgsECI5.png)

## Inference

To do inference, run the following command:

```bash
python inference.py
```

Since the model is trained on [paul-graham-essays](https://huggingface.co/datasets/sgoel9/paul_graham_essays), it will generate text mostly related to business, startups and founders. The sample output is from the same.

Sample Output:

```
Initial text:> Being a founder

Generated Text:
--------------------------------------------------
Being a founder he'll consider himself lucky they're asking for.

Angels are a bit better, but they're metafacts like VCs used to do more than angels and VCs get deals almost exclusively through personal introductions.

The reason VCs want a strong brand is not to draw in more business plans over the transom, but so they win deals when competing against other VCs. Whereas angels are rarely in direct competition, because
--------------------------------------------------
```

## Citation

```bibtex
@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya and others},
  journal={OpenAI blog},
  volume={1},
  number={8},
  pages={9},
  year={2019}
}
```
