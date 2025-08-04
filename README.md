# disaster-tweet-classifier
A Natural Language Processing project to classify tweets as disaster-related or not using the Kaggle “NLP with Disaster Tweets” dataset. Achieved 80.23% accuracy using TF-IDF vectorization and Logistic Regression, with additional metadata features like text length and domain.


# Disaster Tweet Classifier - NLP (80.2% accuracy)

This project solves the [Kaggle Natural Language Processing with Disaster Tweets competition](https://www.kaggle.com/competitions/nlp-getting-started) using TF-IDF vectorization with Logistic Regression and simple but effective metadata features to achieve an accuracy of 80.23%.


## 📌 Objective

The task is to build a model that can automatically determine whether a tweet refers to a real disaster (like an earthquake or hurricane) or is unrelated or figurative (e.g., “this exam is a disaster”).

---

## What I did
1. Preprocessing
    - Removed irrelevant columns like id
  - Cleaned tweet text (lowercasing, punctuation removal, etc.)
  - Created new features:
      - text_len: number of characters in the tweet
      - domain_encoded: encoded domain
        
2. Feature Extraction
  - Applied TF-IDF Vectorization using TfidfVectorizer:
      - ngram_range=(1,2) to include unigrams and bigrams
      - max_features=10,000 and English stop word removal

3. Feature Combination
  - Combined sparse TF-IDF matrix with:
      - text_len and domain_encoded features
  - Used scipy.sparse.hstack to build final feature matrices

4. Modeling
  - Trained a Logistic Regression model with max_iter=200
  - Evaluated performance using:
      - Accuracy
      - Classification report (precision, recall, F1-score)

5. Submission
  - Predicted on the test set
  - Exported predictions in Kaggle submission format

---

## Dataset

The dataset includes:

  - train.csv: Labeled tweets with target = 1 (disaster) or 0 (not disaster)
  - test.csv: Unlabeled tweets for prediction
  - sample_submission.csv: Format for Kaggle submission

Each tweet includes:

  - text: the raw tweet
  - keyword: a keyword from the tweet (if available)
  - location: user-reported location

---

## Tools used

  - Python
  - Pandas, NumPy
  - Scikit-learn
  - SciPy
  - TF-IDF Vectorization (sklearn)
  - Logistic Regression (sklearn)
  - Jupyter Notebook / Kaggle Notebook

---

## Result

  - Accuracy on training set: ~80.23%
  - Strong baseline for further experimentation

---

## Possible Improvements

  - Use cross-validation for more robust evaluation
  - Try advanced models (e.g., LightGBM, BERT)
  - Handle missing values in location more carefully
  - Add sentiment scores or POS tagging as features

---

## 🔗 Kaggle Notebook
[👉 View the Kaggle notebook here](https://www.kaggle.com/code/aayb10/tf-idf-log-reg-for-tweet-classification-80-2)
