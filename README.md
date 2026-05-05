# 🤖 Extractive Question Answering System (NLP)

This project implements a **state-of-the-art Extractive Question Answering (QA) system** using a pre-trained transformer model. It allows users to input a context paragraph and a question, and returns the exact answer span from the text.

The system also includes a **live evaluation module** using the SQuAD v2 dataset and an interactive Flask web interface.

---

## 📖 Overview

The goal of this project is to build a QA system that:

* Takes a **context paragraph**
* Accepts a **natural language question**
* Extracts the **exact answer span** from the context

This is achieved using a **pre-trained transformer model** without additional training.

---

## 🎯 Features

* ✅ Real-time question answering
* ✅ Answer span highlighting in context
* ✅ Confidence score for predictions
* ✅ Evaluation on SQuAD v2 dataset
* ✅ Exact Match (EM) metric calculation
* ✅ Answerability detection accuracy
* ✅ Confusion matrix visualization
* ✅ Live progress streaming (Server-Sent Events)

---

## 📂 Dataset

* **Dataset:** SQuAD v2 (Stanford Question Answering Dataset)
* ~130,000+ QA pairs
* Includes:

  * Answerable questions
  * Unanswerable questions

### Evaluation Setup:

* First **20 validation samples**
* Mixed answerable + unanswerable cases

---

## 🧠 Model Used

### 🔹 Model: `deepset/roberta-base-squad2`

* Base: RoBERTa-base (125M parameters)
* Fine-tuned on: SQuAD v2
* Task: Extractive Question Answering

### How it Works:

* Predicts **start and end positions** of answer in context
* Returns the **highest probability span**
* Outputs confidence score

---

## ⚙️ Methodology

### 🔹 Pipeline 1 — Interactive QA

* User inputs context + question
* Model extracts answer instantly
* Answer is highlighted in the text

### 🔹 Pipeline 2 — Evaluation

* Runs model on dataset samples
* Computes:

  * Exact Match (EM)
  * Answerability Accuracy
* Displays confusion matrix

---

## 📊 Evaluation Metrics

* **Exact Match (EM):**

  * Measures exact string match with ground truth

* **Answerability Accuracy:**

  * Checks if model correctly predicts:

    * Answer exists ✅
    * No answer ❌

* **Confusion Matrix:**

  * TP, TN, FP, FN visualization

---

## 🌐 Web Application

Built using Flask with a responsive UI.

### 🔹 API Routes

* `/` → Main interface
* `/predict` → Get answer from model
* `/evaluate` → Run dataset evaluation

---

### 🔹 Frontend Features

* Context + question input
* Highlighted answer output
* Confidence score display
* Live evaluation progress bar
* Metrics dashboard
* Confusion matrix visualization

---

## 🛠️ Tech Stack

* Python
* Flask
* Hugging Face Transformers
* Hugging Face Datasets
* PyTorch
* Scikit-learn
* HTML / CSS / JavaScript

---

