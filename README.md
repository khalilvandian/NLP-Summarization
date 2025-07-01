# 📄 Abstractive Summarization on XSum

> Comparative Insights into Model Training Strategies  
> Master’s Degree in Data Science – NLP Project  
> Authors: Robin Smith, Babak Khalilvandian, Sergio Verga  
> Institution: Università degli Studi di Milano-Bicocca  

---

## 🔍 Project Overview

This project benchmarks multiple approaches to **abstractive summarization** on the challenging [XSum dataset](https://huggingface.co/datasets/xsum), which pairs BBC news articles with **highly compressed, single-sentence summaries**.

We explore a wide range of modeling strategies, from classical sequence-to-sequence models to advanced transformer architectures, local LLMs, and **parameter-efficient fine-tuning (PEFT)** methods.

---

## 🎯 Goals

- Compare summarization performance across:
  - Classical GRU + Attention baselines
  - Transformer models (T5-small, Flan-T5-base)
  - Local LLMs (LLaMA 3.2:1b, Qwen3:8b via Ollama)
- Evaluate **zero-shot, one-shot, few-shot prompting**
- Fine-tune T5-small using **PEFT methods**: LoRA, Prefix Tuning, IA³, Adapters
- Analyze outputs using **ROUGE** and **BERTScore**
- Apply explainability techniques (Input × Gradient) to interpret model focus

---

## 📁 Dataset

**XSum – Extreme Summarization (BBC News)**  
- 216,511 samples total
- Train/Val/Test split: 204k / 11k / 11k
- Summaries are single-sentence, abstractive, and require deep semantic understanding

**Preprocessing Steps**:
- Removal of duplicates and erroneous entries (e.g., summary longer than document)
- Length-controlled generation based on summary/document ratio

---

## 🧠 Model Architectures

- **GRU Seq2Seq + Luong attention** (baseline)
- **Transformer-based models**:
  - T5-small
  - FLAN-T5-base
- **Local LLMs via Ollama**:
  - LLaMA 3.2:1b
  - Qwen3:8b

---

## 🧪 Fine-Tuning & Prompting

| Strategy        | Description |
|----------------|-------------|
| **Prompting**  | Zero-shot, One-shot, Few-shot (T5/FLAN) |
| **Full FT**    | Update all T5-small parameters |
| **PEFT**       | LoRA, Prefix Tuning, Adapters, IA³ |

---

## 🧾 Decoding Strategies

- **Greedy decoding**
- **Top-k sampling (k = 50)**
- **Top-p (nucleus) sampling (p = 0.9)**
- **Beam Search** (beam size = 4–5)

---

## 📊 Evaluation Metrics

- **ROUGE-1, ROUGE-2, ROUGE-L**: N-gram overlap and LCS
- **BERTScore (F1)**: Semantic similarity using contextual embeddings  
  *(Rescaled with baseline for fairness)*

---

## 📈 Results Summary

| Model                                | ROUGE-1  | ROUGE-2  | ROUGE-L  | BERTScore F1 |
|-------------------------------------|----------|----------|----------|---------------|
| google/flan-t5-base Zero-shot       | 0.338012 | 0.118883 | 0.266840 | 0.392381      |
| google/flan-t5-base One-shot        | 0.337965 | 0.119797 | 0.267914 | **0.395298**  |
| google/flan-t5-base Few-shot        | 0.337772 | **0.119434** | **0.268080** | 0.394133      |
| llama3.2:1b (Ollama)                | 0.231459 | 0.046637 | 0.156299 | 0.228511      |
| Qwen3:8b (Ollama)                   | 0.219483 | 0.056042 | 0.167671 | 0.219449      |
| T5-small Zero Shot                  | 0.171081 | 0.022468 | 0.120879 | 0.088924      |
| T5-small finetuned Zero-shot        | 0.225432 | 0.053187 | 0.174207 | 0.147041      |
| GRU\_pred\_greedy                   | 0.189852 | 0.029122 | 0.139330 | 0.135189      |
| GRU\_pred\_top\_k                   | 0.147645 | 0.013676 | 0.108129 | 0.049020      |
| GRU\_pred\_top\_p                   | 0.137238 | 0.012181 | 0.102272 | 0.034133      |
| GRU\_pred\_beam                     | 0.189561 | 0.029555 | 0.139431 | 0.138498      |
| T5-small\_full\_finetuned greedy    | **0.267096** | 0.068050 | 0.202963 | 0.274889      |
| T5-small\_full\_finetuned top\_k    | 0.235503 | 0.047575 | 0.173452 | 0.222997      |
| T5-small\_full\_finetuned top\_p    | 0.246502 | 0.053227 | 0.182996 | 0.244767      |
| T5-small\_full\_finetuned beam      | 0.252725 | **0.064493** | 0.192806 | 0.253305      |
| T5-small\_ia3 greedy                | 0.005805 | 0.000791 | 0.004120 | -0.475910     |
| T5-small\_ia3 top\_k                | 0.088197 | 0.010947 | 0.063451 | -0.221529     |
| T5-small\_ia3 top\_p                | 0.032910 | 0.004364 | 0.023004 | -0.399074     |
| T5-small\_ia3 beam                  | 0.018928 | 0.002676 | 0.013046 | -0.440056     |
| T5-small\_adapter greedy            | 0.182051 | 0.026008 | 0.127902 | 0.116483      |
| T5-small\_adapter top\_k            | 0.180185 | 0.023186 | 0.122361 | 0.096015      |
| T5-small\_adapter top\_p            | 0.184820 | 0.025604 | 0.126564 | 0.109817      |
| T5-small\_adapter beam              | 0.182140 | 0.026076 | 0.125912 | 0.111487      |
| T5-small\_lora greedy               | 0.201457 | 0.040417 | 0.153050 | -0.414589     |
| T5-small\_lora top\_k               | 0.192382 | 0.027305 | 0.142211 | 0.160064      |
| T5-small\_lora top\_p               | 0.204280 | 0.031014 | 0.150524 | 0.182888      |
| T5-small\_lora beam                 | 0.213166 | 0.043269 | 0.159973 | 0.149846      |
| T5-small\_prefix greedy             | 0.157360 | 0.024807 | 0.123434 | -0.066709     |
| T5-small\_prefix top\_k             | 0.127142 | 0.012411 | 0.096801 | -0.088708     |
| T5-small\_prefix top\_p             | 0.134681 | 0.013950 | 0.102743 | -0.082276     |
| T5-small\_prefix beam               | 0.130399 | 0.024564 | 0.109902 | -0.049091     |

---

## 🧠 Explainability

Used **Inseq** with **Input × Gradient (IxG)** to visualize token-level attribution in summaries.

**Key Findings**:
- Model focuses strongly on main document content, not prompt examples
- Prompting style (0, 1, few-shot) had minimal effect on attention focus

---

## 📌 Key Takeaways

- **FLAN-T5** is the strongest out-of-the-box summarizer
- **Full fine-tuning + LoRA** improves T5-small considerably
- **Qwen3:8b** and **LLaMA** deliver strong semantic output despite lower ROUGE
- Explainability confirms FLAN-T5 is robust to prompt variations
