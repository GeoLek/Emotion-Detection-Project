import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Function to calculate accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Load the tokenizer and the model
model_path = '/home/orion/Geo/Projects/Emotion-Detection-Project'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Check if CUDA (GPU support) is available and use it, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move the model to the specified device

# Load the test data
df_test = pd.read_csv('processed_testdata.csv')

# Map sentiment to numeric labels (assumes sentiments are 'pos' and 'neg')
label_map = {'pos': 1, 'neg': 0}
df_test['label'] = df_test['sentiment'].map(label_map)

# Tokenize and encode the test dataset
encoded_data_test = (tokenizer.batch_encode_plus(
    df_test['processed_text'].values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors='pt'
))

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(df_test['label'].values)

# Create DataLoader
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=16)

# Tracking variables
predictions, true_labels = [], []

# Predict
for batch in dataloader_test:
    batch = tuple(b.to(device) for b in batch)
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = batch[2].to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)

# Calculate performance metrics
accuracy = flat_accuracy(predictions, true_labels)
report = classification_report(true_labels, np.argmax(predictions, axis=1))
conf_matrix = confusion_matrix(true_labels, np.argmax(predictions, axis=1))

print(f'Accuracy: {accuracy}')
print(f'Classification Report: \n{report}')
print(f'Confusion Matrix: \n{conf_matrix}')

# Save the results in a text file
with open('model_performance.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy}\n')
    file.write(f'Classification Report: \n{report}\n')
    file.write(f'Confusion Matrix: \n{conf_matrix}\n')