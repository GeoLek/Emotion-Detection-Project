import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import get_linear_schedule_with_warmup
import random
import os


# Function to set seed for reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


# Set seed for reproducibility
set_seed(42)

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
df = pd.read_csv('processed_imdb_dataset.csv')

# Map sentiments to numerical labels
df['label'] = df['sentiment'].map({'pos': 1, 'neg': 0})

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize and encode the dataset
encoded_data = tokenizer.batch_encode_plus(
    df['processed_text'].values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors='pt'
)

# Extract inputs and labels
input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
labels = torch.tensor(df['label'].values)

# Train-validation split
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    input_ids, labels, random_state=42, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(
    attention_masks, labels, random_state=42, test_size=0.1)

# Create DataLoaders
batch_size = 16

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

# Send model to device
model.to(device)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

# Number of training epochs
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs]
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        batch = tuple(b.to(device) for b in batch)
        b_input_ids, b_input_mask, b_labels = batch

        model.zero_grad()

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()

        # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch} - Average Train Loss: {avg_train_loss}")

# Validation loop
model.eval()

# Tracking variables
val_accuracy = []

for batch in validation_dataloader:
    batch = tuple(b.to(device) for b in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Calculate the accuracy for this batch
    batch_accuracy = accuracy_score(np.argmax(logits, axis=1).flatten(), label_ids.flatten())

    # Accumulate the total accuracy
    val_accuracy.append(batch_accuracy)

# Calculate the average accuracy over all batches
average_accuracy = np.mean(val_accuracy)
print(f"Validation Accuracy: {average_accuracy}")

# Save the fine-tuned model and tokenizer
model_save_path = '/home/orion/Geo/Projects/Emotion-Detection-Project'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("Training complete!")
