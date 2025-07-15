# Sentiment-Analysis-on-Social-Media-Using-Fine-Tuned-BERT-Models
Sentiment Analysis 


# Sentiment Analysis on Social Media Using Fine-Tuned BERT Models

## Overview

This project demonstrates the application of a fine-tuned BERT model for sentiment classification on social media text. It uses a small sample of tweets (Appendix A) and visualizes training performance metrics (Appendix C). The model was developed and evaluated in Python using the Hugging Face Transformers library.

---

## Contents

- `sample_tweets.csv`: Sample tweet dataset with labeled sentiments (0 = Negative, 1 = Neutral, 2 = Positive).
- `accuracy_loss_curves.png`: Visual representation of training/validation accuracy and loss over 3 epochs.
- `appendix_b_code.py`: Full Python script used to generate the dataset and plot the learning curves.
- `appendix_b_code.zip`: Compressed file containing all the above files.
- `README.md`: Project documentation and instructions.

---

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib

Optional for model training:
- torch
- transformers
- datasets
- scikit-learn

