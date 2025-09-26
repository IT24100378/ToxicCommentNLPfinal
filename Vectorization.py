import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


# Load Lemmatized Dataset

df = pd.read_csv("preprocessed_dataset.csv")   # columns: comment, toxic

print("\nDataset Info:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())


# EDA Before Vectorization

def text_eda(data, column_name, title_prefix=""):
    data["word_count"] = data[column_name].apply(lambda x: len(str(x).split()))

    # Word count distribution
    plt.figure(figsize=(6,4))
    sns.histplot(data["word_count"], bins=50, kde=True)
    plt.title(f"{title_prefix} Word Count Distribution")
    plt.show()

    # Top 10 most common words
    all_words = " ".join(data[column_name].astype(str)).split()
    common_words = Counter(all_words).most_common(10)
    common_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])

    plt.figure(figsize=(8,4))
    sns.barplot(x="Frequency", y="Word", data=common_df)
    plt.title(f"{title_prefix} Top 10 Most Common Words")
    plt.show()

    return common_df

print("\n Before Vectorization Analysis:")
top_words_before = text_eda(df.copy(), "comment", title_prefix="Before Vectorization")


#Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["comment"].astype(str))

# Convert TF-IDF matrix into DataFrame
vectorized_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
vectorized_df["toxic"] = df["toxic"]

print(" Data has been vectorized with TF-IDF (5000 features).")


# EDA After Vectorization
# Average TF-IDF for each feature
mean_tfidf = X.mean(axis=0).A1
top_indices = mean_tfidf.argsort()[-10:][::-1]
top_features = [(vectorizer.get_feature_names_out()[i], mean_tfidf[i]) for i in top_indices]

top_df_after = pd.DataFrame(top_features, columns=["Feature", "Average TF-IDF"])

plt.figure(figsize=(8,4))
sns.barplot(x="Average TF-IDF", y="Feature", data=top_df_after)
plt.title("After Vectorization: Top 10 Features by Average TF-IDF Weight")
plt.show()


#Compare Before vs After

comparison_df = pd.DataFrame({
    "Top Words Before": top_words_before["Word"],
    "Freq Before": top_words_before["Frequency"],
    "Top Features After": top_df_after["Feature"],
    "Avg TF-IDF After": top_df_after["Average TF-IDF"]
})

print("\n Comparison of Top Features Before vs After Vectorization:\n")
print(comparison_df)


# Save Final Preprocessed Dataset

vectorized_df.to_csv("final_preprocessed.csv", index=False)
print("Final preprocessed dataset saved as final_preprocessed.csv")
