import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load your fine-tuned BERT model and tokenizer
model_directory = '/home/orion/Geo/Projects/Emotion-Detection-Project'  # Replace with your model directory
tokenizer = BertTokenizer.from_pretrained(model_directory)
model = BertForSequenceClassification.from_pretrained(model_directory)

# Create a linguistic variable for sentiment based on text analysis
sentiment = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'sentiment')

# Create an output variable for sentiment
output = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'output')

# Adjusted linguistic terms for sentiment
sentiment['negative'] = fuzz.trimf(sentiment.universe, [0, 0, 0.1])
sentiment['positive'] = fuzz.trimf(sentiment.universe, [0.1, 1, 1])

# Adjusted output membership functions for "negative" and "positive"
output['negative'] = fuzz.trimf(output.universe, [0, 0, 0.1])
output['positive'] = fuzz.trimf(output.universe, [0.1, 1, 1])

# Adjusted fuzzy rules for sentiment classification
rule1 = ctrl.Rule(sentiment['negative'], output['negative'])
rule2 = ctrl.Rule(sentiment['positive'], output['positive'])

# Create a control system for sentiment analysis with the adjusted rules
sentiment_ctrl = ctrl.ControlSystem([rule1, rule2])
sentiment_output = ctrl.ControlSystemSimulation(sentiment_ctrl)

# Function to predict sentiment using fuzzy logic based on text analysis
def predict_sentiment_fuzzy(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

    # Print Tokenized Input
    print("Tokenized Input:", inputs)

    # Make the prediction using the BERT model
    with torch.no_grad():
        outputs = model(**inputs)

    # Print BERT Model Output (Raw Scores)
    print("BERT Model Output (Raw Scores):", outputs.logits)

    # Extract the sentiment score (assumes higher scores are more positive)
    sentiment_score = torch.sigmoid(outputs.logits)[0][0].item()  # Extract and convert to scalar

    # Print Fuzzified Sentiment Score
    print("Fuzzified Sentiment Score:", sentiment_score)

    # Set the sentiment input value in the fuzzy system
    sentiment_output.input['sentiment'] = sentiment_score

    # Compute the fuzzy logic output
    sentiment_output.compute()

    # Get the predicted sentiment (output) from the fuzzy system
    predicted_sentiment = sentiment_output.output['output'].item()  # Extract and convert to scalar

    # Determine whether it's positive or negative based on the output value
    if predicted_sentiment <= 0.60:
        return "Positive"
    else:
        return "Negative"

# Example usage:
input_text = "I was so moved by this film in 1981, I went back to the theater four times to see it again! Something I have never done for another film. No movie evokes the feelings of growing up in the 60's like Four Friends. That it so closely approximated my own experiences in the 60's is probably something that many will share. Jodi Thelen is radiantly beautiful and unforgetable! Why she didn't become a major star after this I will never know. The acting by the entire cast is flawless as is Steve Tisch's script. I always wanted to know how much of the story was autobiographical. But alas, Steve is no longer here to answer that question. I have all but worn out my VHS copy of this great movie! Highly recommended!"

predicted_fuzzy_sentiment = predict_sentiment_fuzzy(input_text)

print("Predicted Sentiment:", predicted_fuzzy_sentiment)
