import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('/home/orion/Geo/Projects/Emotion-Detection-Project/bert_features_test.csv')

# Define the smaller range intervals of 0.01
range_step = 0.01
min_range = -3.5
max_range = 3.5
ranges = [(lower, lower + range_step) for lower in np.arange(min_range, max_range, range_step)]

# Initialize counters for positive and negative sentiment
positive_counts = [0] * len(ranges)
negative_counts = [0] * len(ranges)

# Count the values within each range for positive sentiment
for col in df.columns[2:]:
    values = df.iloc[1:12501][col]  # Rows 2 to 12501 for positive sentiment
    for i, (lower, upper) in enumerate(ranges):
        count = values.between(lower, upper).sum()
        positive_counts[i] += count

# Count the values within each range for negative sentiment
for col in df.columns[2:]:
    values = df.iloc[12502:25001][col]  # Rows 12502 to 25001 for negative sentiment
    for i, (lower, upper) in enumerate(ranges):
        count = values.between(lower, upper).sum()
        negative_counts[i] += count

# Filter out ranges with count 0
non_zero_ranges = []
non_zero_positive_counts = []
non_zero_negative_counts = []

for i, count in enumerate(positive_counts):
    if count > 0:
        non_zero_ranges.append(ranges[i])
        non_zero_positive_counts.append(count)
        non_zero_negative_counts.append(negative_counts[i])

# Save the ranges and counts for positive sentiment to a text file
with open('positive_sentiment_ranges.txt', 'w') as file:
    for (lower, upper), count in zip(non_zero_ranges, non_zero_positive_counts):
        file.write(f'Range {lower}-{upper}: Count {count}\n')

# Save the ranges and counts for negative sentiment to a text file
with open('negative_sentiment_ranges.txt', 'w') as file:
    for (lower, upper), count in zip(non_zero_ranges, non_zero_negative_counts):
        file.write(f'Range {lower}-{upper}: Count {count}\n')

# Create a comprehensive plot for positive sentiment
plt.figure(figsize=(12, 6))

# Plot positive sentiment counts
plt.plot([f'{lower:.2f}-{upper:.2f}' for lower, upper in non_zero_ranges], non_zero_positive_counts, label='Positive Sentiment', marker='o')

# Configure the plot for positive sentiment
plt.xlabel('Value Ranges')
plt.ylabel('Count')
plt.title('Positive Sentiment Analysis by Value Ranges (Excluding 0 Counts)')
plt.xticks(rotation=90)
plt.legend()

# Save the plot for positive sentiment
plt.tight_layout()
plt.savefig('positive_sentiment_plot_non_zero.png')
plt.close()

# Create a comprehensive plot for negative sentiment
plt.figure(figsize=(12, 6))

# Plot negative sentiment counts
plt.plot([f'{lower:.2f}-{upper:.2f}' for lower, upper in non_zero_ranges], non_zero_negative_counts, label='Negative Sentiment', marker='x')

# Configure the plot for negative sentiment
plt.xlabel('Value Ranges')
plt.ylabel('Count')
plt.title('Negative Sentiment Analysis by Value Ranges (Excluding 0 Counts)')
plt.xticks(rotation=90)
plt.legend()

# Save the plot for negative sentiment
plt.tight_layout()
plt.savefig('negative_sentiment_plot_non_zero.png')
plt.close()
