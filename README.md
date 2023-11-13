# Disinformation Capabilities of Large Language Models

This is the source code for the paper Disinformation Capabilities of Large Language Models.

## Abstract

Automated disinformation generation is often listed as one of the risks of large language models (LLMs). The theoretical ability to flood the information space with disinformation content might have dramatic consequences for democratic societies around the world. This paper presents a comprehensive study of the disinformation capabilities of the current generation of LLMs to generate false news articles in English language. In our study, we evaluated the capabilities of 10 LLMs using 20 disinformation narratives. We evaluated several aspects of the LLMs: how well they are at generating news articles, how strongly they tend to agree or disagree with the disinformation narratives, how often they generate safety warnings, etc. We also evaluated the abilities of detection models to detect these articles as LLM-generated. We conclude that LLMs are able to generate convincing news articles that agree with dangerous disinformation narratives.

## Project Structure

- `src/detection`: Run detection methods.
- `src/evaluation`: Evaluate detection methods and human annotation.
- `src/generation`: Source code for generating disinformation news articles.

## News Articles Generation

### Prompts Creation

To create a list of prompts, execute the code in `src/generation/Prompts creation.ipynb`. This Jupyter notebook generates `prompts.csv` in the `data` folder, which is used to generate disinformation news articles.

### Generation of Texts

After preparing prompts, text generation can commence. Utilize the OpenAI API for text generation by running `src/generation/Batch generation.ipynb`. For other models, use the provided Python scripts (`infer_falcon-40b-instruct.py`, `infer_llama2-70b-instruct.py`, `infer_mistral7b.py`, `infer_opt-iml-max-30b.py`, or `infer_vicuna33b.py`) from the `src/generation` folder, for example:

```bash
python infer_vicuna33b.py
```

## Detection methods

### Data preparation

Create `articles.csv` in the `data` folder with a single column `article` to prepare data. Merge the generated text with human-written data for the detection experiment using `Prepare data.ipynb`.

### Running detection methods

Select a specific detection method or run all methods (using `--model all`). Execute the detection script from the `src/detection`:

```bash
python script.py --model all --model_path ../../data/models
```

After running detection, find the results in `src/data/results`.

## Paper citing

If you use the data, code, or information from this repository, please cite our paper, also available on [arXiv](link_to_arxiv).

```bibtex
@misc{
  vykopal2023disinformation,
  title={Disinformation Capabilities of Language Models},
  author={Ivan Vykopal and Matúš Pikuliak and Ivan Srba and Robert Moro and Dominik Macko and Maria Bielikova},
  year={2023},
  archivePrefix={arXiv},
  eprint={insert_eprint_number},
  url={link_to_paper}
}
```