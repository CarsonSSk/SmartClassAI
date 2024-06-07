import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os

base_dir="models"
model_name="model_2024-06-06_02-31-02"
# Load the results CSV file
results_df = pd.read_csv(os.path.join(os.path.join(base_dir, model_name),"results.csv"))

# Convert the labels to numerical values
label_mapping = {'happy': 0, 'angry': 1, 'neutral': 2, 'engaged': 3}
results_df['Correct Label'] = results_df['Correct Label'].map(label_mapping)
results_df['Predicted Label'] = results_df['Predicted Label'].map(label_mapping)

# Calculate evaluation metrics
micro_precision = precision_score(results_df['Correct Label'], results_df['Predicted Label'], average='micro')
micro_recall = recall_score(results_df['Correct Label'], results_df['Predicted Label'], average='micro')
micro_f1 = f1_score(results_df['Correct Label'], results_df['Predicted Label'], average='micro')

macro_precision = precision_score(results_df['Correct Label'], results_df['Predicted Label'], average='macro')
macro_recall = recall_score(results_df['Correct Label'], results_df['Predicted Label'], average='macro')
macro_f1 = f1_score(results_df['Correct Label'], results_df['Predicted Label'], average='macro')

accuracy = accuracy_score(results_df['Correct Label'], results_df['Predicted Label'])

# Print the evaluation metrics
print(f"Micro-averaged Precision: {micro_precision:.4f}")
print(f"Micro-averaged Recall: {micro_recall:.4f}")
print(f"Micro-averaged F1 Score: {micro_f1:.4f}")
print(f"Macro-averaged Precision: {macro_precision:.4f}")
print(f"Macro-averaged Recall: {macro_recall:.4f}")
print(f"Macro-averaged F1 Score: {macro_f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
