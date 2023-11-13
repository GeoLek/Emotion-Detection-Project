import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
df = pd.read_csv('processed_imdb_dataset.csv')

# Class Distribution
# This plot will show the number of instances for each sentiment class.
sns.countplot(x='sentiment', data=df)
plt.title('Class Distribution')
plt.savefig('class_distribution.png')  # Save the figure
plt.clf()  # Clear the current figure

# Text Length Analysis
# Here, we are calculating the length of each text entry and plotting its distribution.
df['text_length'] = df['text'].apply(len)
sns.histplot(x='text_length', data=df, bins=50, kde=True)
plt.title('Text Length Distribution')
plt.savefig('text_length_distribution.png')  # Save the figure
plt.clf()  # Clear the current figure

# Average Text Length per Sentiment
# We calculate and print the average length of text for each sentiment.
avg_text_length = df.groupby('sentiment')['text_length'].mean()
print("Average text length per sentiment:\n", avg_text_length)

# Word Frequency Analysis for 'processed_text'
# Count the most common words in the 'processed_text' column.
all_words = ' '.join(df['processed_text']).split()
word_freq = Counter(all_words)
most_common_words = word_freq.most_common(20)
print("Most common words:\n", most_common_words)

# Word Clouds
# Generate and save a word cloud image for the processed text.
wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(' '.join(df['processed_text']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Word Cloud for Processed Text')
plt.savefig('wordcloud.png')  # Save the word cloud image
plt.clf()  # Clear the current figure

# N-Grams Analysis (Example for Bi-grams)
# Define a function to get the most common N-grams.
def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:10]

# Get and print the top bi-grams
top_bi_grams = get_top_ngram(df['processed_text'], 2)
print("Top bi-grams:\n", top_bi_grams)

# Save the results of the analysis to a text file
with open('eda_results.txt', 'w') as f:
    f.write(f"Average text length per sentiment:\n{avg_text_length}\n\n")
    f.write(f"Most common words:\n{most_common_words}\n\n")
    f.write(f"Top bi-grams:\n{top_bi_grams}\n")
