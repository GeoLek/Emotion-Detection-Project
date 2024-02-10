from memory_profiler import profile
import pandas as pd
import numpy as np
import torch
import gc  # Import garbage collector
from transformers import BertTokenizer, BertModel
from sklearn import preprocessing

# Define the path to your fine-tuned BERT model and your datasets
model_directory = '/home/orion/Geo/Projects/Emotion-Detection-Project'
train_data_path = '/home/orion/Geo/Projects/Emotion-Detection-Project/processed_imdb_dataset.csv'
test_data_path = '/home/orion/Geo/Projects/Emotion-Detection-Project/processed_testdata.csv'

# Load the tokenizer and model from the fine-tuned BERT
tokenizer = BertTokenizer.from_pretrained(model_directory)
model = BertModel.from_pretrained(model_directory)

# Ensure the model is in evaluation mode
model.eval()

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

@profile
def process_batch(batch_texts, tokenizer, model):
    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    gc.collect()  # Prompt garbage collection
    return outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

@profile
def extract_and_normalize_features(filepath, tokenizer, model, text_column, batch_size=32):
    features = []
    for chunk in pd.read_csv(filepath, chunksize=batch_size):
        batch_features = process_batch(chunk[text_column].tolist(), tokenizer, model)
        features.append(batch_features)
        gc.collect()  # Prompt garbage collection after processing each chunk

    features_array = np.vstack(features)
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(features_array)

@profile
def main():
    train_features = extract_and_normalize_features(train_data_path, tokenizer, model, 'processed_text', batch_size=32)
    train_features_df = pd.DataFrame(train_features, columns=[f'bert_feature_{i}' for i in range(train_features.shape[1])])
    train_features_df.insert(0, 'id', pd.read_csv(train_data_path)['id'])
    train_features_df.insert(1, 'sentiment', pd.read_csv(train_data_path)['sentiment'])
    train_features_df.to_csv('bert_features_train.csv', index=False)

    test_features = extract_and_normalize_features(test_data_path, tokenizer, model, 'processed_text', batch_size=32)
    test_features_df = pd.DataFrame(test_features, columns=[f'bert_feature_{i}' for i in range(test_features.shape[1])])
    test_features_df.insert(0, 'id', pd.read_csv(test_data_path)['id'])
    test_features_df.insert(1, 'sentiment', pd.read_csv(test_data_path)['sentiment'])
    test_features_df.to_csv('bert_features_test.csv', index=False)

    print("Feature extraction and saving complete.")

if __name__ == "__main__":
    main()