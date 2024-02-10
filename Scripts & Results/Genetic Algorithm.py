import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from deap import base, creator, tools, algorithms
import random
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load your fine-tuned BERT model and tokenizer
model_directory = '/home/orion/Geo/Projects/Emotion-Detection-Project'
tokenizer = BertTokenizer.from_pretrained(model_directory)
model = BertForSequenceClassification.from_pretrained(model_directory)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
NUM_PARAMS = 3
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NUM_PARAMS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def load_test_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    test_data = [(row['text'], "Positive" if row['sentiment'] == 'pos' else "Negative") for index, row in df.iterrows()]
    return test_data

def evaluate(individual, test_data):
    accuracy, precision, recall, f1 = run_fuzzy_system(individual, test_data)
    return (accuracy,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def create_fuzzy_system(params):
    params = np.clip(params, 0, 1)
    sentiment = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'sentiment')
    output = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'output')

    sentiment['negative'] = fuzz.trimf(sentiment.universe, [0, 0, params[0]])
    sentiment['positive'] = fuzz.trimf(sentiment.universe, [params[1], 1, 1])
    output['negative'] = fuzz.trimf(output.universe, [0, 0, params[2]])
    output['positive'] = fuzz.trimf(output.universe, [params[2], 1, 1])

    rule1 = ctrl.Rule(sentiment['negative'], output['negative'])
    rule2 = ctrl.Rule(sentiment['positive'], output['positive'])

    sentiment_ctrl = ctrl.ControlSystem([rule1, rule2])
    sentiment_output = ctrl.ControlSystemSimulation(sentiment_ctrl)
    return sentiment_ctrl, sentiment_output

def evaluate_system(sentiment_ctrl, sentiment_output, test_data):
    predictions, true_labels = [], []
    for input_text, true_label in test_data:
        predicted_label = predict_sentiment_fuzzy(input_text, sentiment_ctrl, sentiment_output)
        predictions.append(predicted_label)
        true_labels.append(true_label)
    # Convert labels to binary format for sklearn metrics
    binary_predictions = [1 if pred == "Positive" else 0 for pred in predictions]
    binary_true_labels = [1 if label == "Positive" else 0 for label in true_labels]
    accuracy = accuracy_score(binary_true_labels, binary_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(binary_true_labels, binary_predictions, average='binary')
    return accuracy, precision, recall, f1

def run_fuzzy_system(individual, test_data):
    sentiment_ctrl, sentiment_output = create_fuzzy_system(individual)
    return evaluate_system(sentiment_ctrl, sentiment_output, test_data)

def predict_sentiment_fuzzy(input_text, sentiment_ctrl, sentiment_output):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    sentiment_score = torch.sigmoid(outputs.logits)[0][0].item()
    sentiment_output.input['sentiment'] = sentiment_score

    try:
        sentiment_output.compute()
        predicted_sentiment = sentiment_output.output['output'].item()
        return "Positive" if predicted_sentiment <= 0.60 else "Negative"
    except ValueError as e:
        print(f"Error for input '{input_text}': {e}")
        # Default or backup logic here
        # For example, return "Unknown" or apply a different logic to decide
        return "Unknown"  # or "Positive" if sentiment_score > threshold else "Negative"

def main(test_data):
    population = toolbox.population(n=50)
    NGEN = 50

    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind, test_data)
        population[:] = toolbox.select(offspring, len(population))

    best_individual = tools.selBest(population, 1)[0]
    best_accuracy, best_precision, best_recall, best_f1 = run_fuzzy_system(best_individual, test_data)
    return best_individual, best_accuracy, best_precision, best_recall, best_f1

if __name__ == "__main__":
    csv_file_path = '/home/orion/Geo/Projects/Emotion-Detection-Project/testdata.csv'  # Update this path
    test_data = load_test_data(csv_file_path)
    best_params, accuracy, precision, recall, f1 = main(test_data)
    print(f"Best Parameters: {best_params}")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Save to text file
    with open('performance_metrics_and_params2.txt', 'w') as file:
        file.write(f"Best Parameters: {best_params}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1 Score: {f1}\n")