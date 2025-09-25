
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# =========================
# STEP 1 - Load the Dataset
# =========================


df = pd.read_csv("data/train.csv")

print("Dataset shape:", df.shape)
print(df.head())