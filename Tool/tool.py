  ########## 1. Import required libraries ##########
import random
import pandas as pd
import numpy as np
import re
import math

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# Text cleaning & stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
########## 2. Define text preprocessing methods ##########


def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_short_words(text):
    return " ".join([w for w in text.split() if len(w) > 2])

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Stopwords
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = []  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

########## 3. Download & read data ##########
import os
import subprocess
# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'tensorflow'
path = f'{project}.csv'

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle

# Merge Title and Body into a single column; if Body is NaN, use Title only
pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

# Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

########## 4. Configure parameters & Start training ##########

# ========== Key Configurations ==========

# 1) Data file to read
datafile = 'Title+Body.csv'

# 2) Number of repeated experiments
REPEAT = 10

# 3) Output CSV file name
out_csv_name = f'../{project}_LinearSVC.csv'

# ========== Read and clean data ==========
data = pd.read_csv(datafile).fillna('')
text_col = 'text'

# Keep a copy for referencing original data if needed
original_data = data.copy()

# Text cleaning
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(remove_short_words)
data[text_col] = data[text_col].apply(lemmatize_text)
data[text_col] = data[text_col].apply(clean_str)

# ========== Hyperparameter grid ==========
# We use logspace for var_smoothing: [1e-12, 1e-11, ..., 1]
params = {
    'C': np.logspace(-12, 0, 13)
}

# Lists to store metrics across repeated runs
accuracies  = []
precisions  = []
recalls     = []
f1_scores   = []
auc_values  = []

# --- 4.1 Split into train/test ---
indices = np.arange(data.shape[0])
train_index, test_index = train_test_split(
    indices, test_size=0.2, random_state=8
)

train_text = data[text_col].iloc[train_index]
test_text = data[text_col].iloc[test_index]

y_train = data['sentiment'].iloc[train_index]
y_test  = data['sentiment'].iloc[test_index]

# --- 4.2 TF-IDF vectorization ---
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=4000 # Adjust as needed
)
X_train = tfidf.fit_transform(train_text)
X_test = tfidf.transform(test_text)

# --- 4.3 Naive Bayes model & GridSearch ---
model = LinearSVC(class_weight='balanced', max_iter=5000)
grid = GridSearchCV(
    model,
    params,
    cv=5,              # 5-fold CV (can be changed)
    scoring='roc_auc'  # Using roc_auc as the metric for selection
)
grid.fit(X_train, y_train)
    
# Retrieve the best model
best_model = grid.best_estimator_
best_model.fit(X_train, y_train)
    
# --- 4.4 Make predictions & evaluate ---
y_pred = best_model.predict(X_test)
# Accuracy
acc = accuracy_score(y_test, y_pred)
accuracies.append(acc)

# Precision (macro)
prec = precision_score(y_test, y_pred, average='macro',zero_division = 0)
precisions.append(prec)

# Recall (macro)
rec = recall_score(y_test, y_pred, average='macro', zero_division = 0)
recalls.append(rec)

# F1 Score (macro)
f1 = f1_score(y_test, y_pred, average='macro', zero_division = 0)
f1_scores.append(f1)

# AUC
# If labels are 0/1 only, this works directly.
# If labels are something else, adjust pos_label accordingly.
fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
auc_val = auc(fpr, tpr)
auc_values.append(auc_val)

# --- 4.5 Aggregate results ---
final_accuracy  = np.mean(accuracies)
final_precision = np.mean(precisions)
final_recall    = np.mean(recalls)
final_f1        = np.mean(f1_scores)
final_auc       = np.mean(auc_values)

print("=== Current Model ===")
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 score:      {final_f1:.4f}")
print(f"Average AUC:           {final_auc:.4f}")



def predict_bug(model=best_model, tfidf=tfidf):
    string = input("Bug report text here:")
    
    string = remove_html(string)
    string = remove_emoji(string)
    string = remove_stopwords(string)
    string = remove_short_words(string)
    string = lemmatize_text(string)
    string = clean_str(string)

    processed_text = tfidf.transform([string]).toarray()

    predict = best_model.predict(processed_text)[0]
    confidence = best_model.decision_function(processed_text)[0]
    print (" ")
    match predict:
        case 1:
            print ("Bug Report is Performance Related. {}".format(confidence))
        case 0:
            print ("Bug Report is not Performance Related. {}".format(confidence))

def predict(input_file,model=best_model, tfidf=tfidf):
    file = pd.read_csv("{}.csv".format(input_file))
    original = file.copy()
    file['Title+Body'] = file.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )
    file_tplusb = file.rename(columns={
        "Unnamed: 0": "id",
        "class": "sentiment",
        "Title+Body": "text"
    }).fillna('')
    data = file_tplusb
    
    data["text"] = data["text"].apply(remove_html)
    data["text"] = data["text"].apply(remove_emoji)
    data["text"] = data["text"].apply(remove_stopwords)
    data["text"] = data["text"].apply(remove_short_words)
    data["text"] = data["text"].apply(lemmatize_text)
    data["text"] = data["text"].apply(clean_str)
    vectors = tfidf.transform(data["text"])

    predictions = model.predict(vectors)
    original["predicted class"] = predictions
    original["label"] = original["predicted class"].apply(label_conversion)
    original.to_csv("{}_predictions.csv".format(input_file))

    print ("Saved to {}_predictions.csv.".format(input_file))    

def label_conversion(x):
    if x:
        return "Performance"
    else:
        return "Not Performance"
















