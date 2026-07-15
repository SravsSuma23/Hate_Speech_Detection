# 🛡️ Scalable Big Data Forensic System for Detecting Cybercrimes on Social Media

## Overview

Social media platforms generate millions of multimedia posts every day, making it increasingly difficult to identify harmful content such as hate speech, cyberbullying, and abusive memes. Traditional text-only or image-only detection systems often fail to capture the complex relationship between visual and textual information embedded in memes.

This project presents a **Scalable Big Data Forensic System** that leverages **Multimodal Deep Learning** and **Big Data technologies** to accurately detect hate speech in social media memes. The system combines **Natural Language Processing (NLP)** and **Computer Vision (CV)** techniques to analyze both textual and visual content simultaneously, providing a more robust and reliable classification.

Designed for large-scale social media analysis, the solution utilizes **Apache Spark** for distributed data processing and **Hadoop Distributed File System (HDFS)** for scalable storage, enabling efficient handling of massive datasets while maintaining high performance.

---

## 🚀 Key Features

* 🔍 **Multimodal Hate Speech Detection** using both image and text information.
* 🤖 **RoBERTa-based NLP pipeline** for extracting contextual textual embeddings.
* 🖼️ **Deep CNN-based image feature extraction** for visual content analysis.
* ⚡ **Apache Spark-powered distributed processing** for scalable big data analytics.
* 💾 **HDFS integration** for fault-tolerant and distributed storage.
* 📈 **SMOTE-based data balancing** to address class imbalance and improve model generalization.
* 🧠 **Deep Learning models** including LSTM and Transformer architectures for accurate classification.
* 📊 Comprehensive preprocessing, feature standardization, model training, and evaluation pipeline.
* 🔄 Modular architecture designed for future deployment and real-time social media monitoring.

---

## 🎯 Problem Statement

Hate speech hidden within memes poses a significant challenge because offensive intent is often conveyed through the combination of images and text rather than either modality alone. Existing single-modal approaches frequently fail to understand this context, resulting in poor detection accuracy.

This project addresses this limitation by developing a scalable multimodal forensic framework capable of identifying harmful meme content with greater accuracy while efficiently processing large-scale datasets.

---

## 🏗️ System Architecture

The complete pipeline consists of the following stages:

1. Data Collection & Preprocessing
2. Image Feature Extraction using Deep Learning
3. Text Feature Extraction using RoBERTa
4. Feature Standardization
5. SMOTE-based Class Balancing
6. Feature Fusion
7. Model Training (LSTM & Transformer)
8. Hate Speech Classification
9. Performance Evaluation

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Big Data Technologies

* Apache Spark
* Hadoop HDFS

### Machine Learning & Deep Learning

* PyTorch
* Transformers (RoBERTa)
* LSTM
* Transformer Networks
* SMOTE

### Data Processing

* Pandas
* NumPy
* Scikit-learn

### Computer Vision

* OpenCV
* PIL

### NLP

* Hugging Face Transformers
* RoBERTa

---

## 📊 Classification Labels

The model classifies every meme into one of the following categories:

* ✅ Non-Hateful
* 🚫 Hateful

---

## 🌟 Project Highlights

* Developed an end-to-end **multimodal AI pipeline** integrating Computer Vision and NLP.
* Built a **scalable architecture** capable of processing large datasets using Apache Spark.
* Improved detection performance through **feature fusion** and **class imbalance handling**.
* Designed with a modular structure, making it suitable for research, further development, and production deployment.

---

## 🎯 Applications

* Social Media Content Moderation
* Cybercrime Investigation
* Digital Forensics
* Online Hate Speech Detection
* AI-powered Trust & Safety Systems
* Research in Responsible AI and Content Moderation

---

## 👩‍💻 Author

**Madhupada Sravanthi Suma**

B.Tech in Artificial Intelligence | Machine Learning | Deep Learning | NLP | Computer Vision | Big Data Analytics

