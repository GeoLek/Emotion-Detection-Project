import os
import pandas as pd

def load_data_from_folder(directory, sentiment):

   # Load and process data from a specific folder.


    rows = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Check if the file is a text format
            # Extract ID and rating from the filename
            id, rating = filename.split('_')
            rating = rating.split('.')[0]

            # Construct the full path to the file
            filepath = os.path.join(directory, filename)

            # Open and read the content of the file
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()

            # Append a dictionary with the file's data to the rows list
            rows.append({'id': id, 'rating': rating, 'sentiment': sentiment, 'text': text})

    return rows

def load_data(base_directory):

   # Load data from both positive and negative folders.

    # Load data from the 'pos' folder, assigning 'pos' as the sentiment label
    pos_data = load_data_from_folder(os.path.join(base_directory, 'pos'), 'pos')

    # Load data from the 'neg' folder, assigning 'neg' as the sentiment label
    neg_data = load_data_from_folder(os.path.join(base_directory, 'neg'), 'neg')

    # Combine data from both folders
    all_data = pos_data + neg_data

    # Convert the combined data into a pandas DataFrame and return it
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    # Define the base directory containing 'pos' and 'neg' subdirectories
    base_dataset_dir = '/home/orion/Geo/Projects/Emotion-Detection-Project/train'

    # Load the data from both subdirectories and create a DataFrame
    df = load_data(base_dataset_dir)

    # Save the DataFrame to a CSV file
    df.to_csv('imdb_dataset.csv', index=False)
