# NLP-course-CIMAT

Author: Miguel Angel Ruiz Ortiz

This repository contains my code for the **Natural Language Processing (NLP) graduate course** at CIMAT (Centro de Investigación en Matemáticas, Mexico) in 2025, taught by Dr. Adrián Pastor López Monroy and Dr. Fernando Sanchez Vega.

The work covers both **traditional NLP techniques** and **deep learning approaches**, with applications to real-world datasets.

---

## Tools & Libraries

Python, PyTorch, TensorFlow/Keras, Hugging Face, Scikit-learn, NLTK, NumPy, SciPy, Matplotlib, Seaborn, Google Colab (GPU), Git/GitHub.

---

## Dataset Sources

- [PAN17 Author Profiling](https://pan.webis.de/clef17/pan17-web/author-profiling.html)
- TripAdvisor reviews: Guanajuato tourist sites reviews dataset.  
- Mexican politics transcripts: speeches by AMLO and Sheinbaum (publicly available online).  

---

## Tasks Overview

### Homework 1 — NLTK Library & Data Exploration

Analysis of AMLO and Sheinbaum speeches using NLTK.

### Homework 2 — Text Mining & BoW Representations

Hate speech detection using Bag-of-Words (binary, frequency, TF-IDF) and SVM classifiers.

### Homework 3 — Distributional Term Representations & Feature Selection

Programmed **TCOR** (Term Co-Occurrence Representation) and **Random Indexing**, explored term similarity and implemented feature selection via **Chi^2**.

### Homework 4 — N-gram Language Models

Built N-gram language models on Mexican politics dataset with interpolation and perplexity analysis.

### Homework 5 — Neural Language Models

Implemented word-level and character-level neural language models (based on Bengio, 2003).

### Homework 6 — Hierarchical Attention Network (HAN)

Developed a **Hierarchical Attention Network** for **author profiling** (nationality) from Spanish tweets.

---

## Exams & Projects

### Test 1 — TripAdvisor Reviews Analysis

Feature selection with **Chi^2** and **Latent Semantic Analysis (LSA)** on Guanajuato tourist site reviews.

### Test 2 — Author Profiling from Spanish Tweets

Implemented a **HAN-based architecture** with **RoBERTuito** (BERT pretrained on Spanish tweets) for **nationality and gender classification**.  
Achieved **83% accuracy**. Used ensemble models for final predictions.

### Final Project — MiniLM: Deep Self-Attention Distillation for Transformers

Studied and presented the paper ["MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-trained Transformers"](https://arxiv.org/abs/2002.10957), by Wenhui Wang et al. (2020). Implemented the distillation technique from scratch and fine-tuned the resulting MiniLM model on **SST-2 binary classification** task.