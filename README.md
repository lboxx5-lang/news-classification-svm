# News Text Classification with TF-IDF and Linear SVM

This repository contains a supervised NLP project for multi-class news/article classification.

The goal is to predict the target class of each article using textual content and metadata features. The final model combines TF-IDF representations, source information, numerical metadata and a class-balanced Linear Support Vector Classifier.

## Project Overview

The pipeline uses:

- article title and body text
- unigram and bigram TF-IDF features
- weighted trigram TF-IDF features
- one-hot encoded source metadata
- page-rank information
- text-length features
- class-balanced LinearSVC classification
- post-hoc calibration of decision scores

The final output is a submission file containing predicted labels for the evaluation set.

## Methods

- Natural Language Processing
- Text Classification
- TF-IDF Vectorization
- N-gram Feature Engineering
- Linear Support Vector Machines
- Metadata Feature Integration
- Post-hoc Score Calibration

## Tools

- Python
- NumPy
- pandas
- scikit-learn
- Jupyter Notebook

## Repository Structure

```text
.
├── src/
│   └── train_predict.py
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── report/
│   └── News_Classification_Report_Luca_Bosso.pdf
├── data/
│   └── evaluation.csv
├── outputs/
│   └── sample_submission.csv
├── requirements.txt
├── .gitignore
└── README.md
```
## Data
Data is included in the "Data" folder inside the evaluation set. The real training set was much bigger but unfortunately github doesn't let me upload over 25MB.

Expected columns:

development.csv
- Id
- title
- article
- source
- page_rank
- label

evaluation.csv

- Id
- title
- article
- source
- page_rank

## How to Run
Install dependencies:
```pip install -r requirements.txt```

Run the training and prediction script:
```python src/train_predict.py```

The script generates:
```outputs/submission.csv```

## Model

The model uses a scikit-learn Pipeline and ColumnTransformer to combine text, categorical and numerical features.

Text features:

- TF-IDF unigrams and bigrams
- weighted TF-IDF trigrams

Metadata features:

- one-hot encoded source
- page rank
- text length

Classifier:

- class-balanced LinearSVC

Final predictions are obtained from calibrated decision scores.

## Report

The technical report is available in:
```report/News_Classification_Report_Luca_Bosso.pdf```

## Skills Demonstrated

- NLP
- Text classification
- Feature engineering
- TF-IDF vectorization
- Linear SVM models
- scikit-learn pipelines
- Data preprocessing
- Model evaluation
- Python machine learning workflow

## Author

Luca Bosso
Graduate student in Data Science and Engineering at Politecnico di Torino
Incoming Master of Science in Computer Science student at University of Illinois Chicago
