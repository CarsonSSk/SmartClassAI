import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import numpy as np

# Specify the model names that you wish to evaluate
main_model_dir = "model_2024-06-24_18-10-47"

variant1_model_dir = "variant1_model_2024-06-15_15-47-03"

variant2_model_dir = "model_2024-06-24_18-10-47"

best_model_dir = "model_2024-06-27_22-56-49"

base_dir = "models"
label_mapping = {'happy': 0, 'angry': 1, 'neutral': 2, 'engaged': 3}
classes = ['happy', 'angry', 'neutral', 'engaged']

def load_results(model_dir):
    results_df = pd.read_csv(os.path.join(base_dir, model_dir, "results.csv"))
    results_df['Correct Label'] = results_df['Correct Label'].map(label_mapping)
    results_df['Predicted Label'] = results_df['Predicted Label'].map(label_mapping)
    return results_df

def evaluate_and_get_metrics(results_df):
    accuracy = accuracy_score(results_df['Correct Label'], results_df['Predicted Label'])
    macro_precision = precision_score(results_df['Correct Label'], results_df['Predicted Label'], average='macro')
    macro_recall = recall_score(results_df['Correct Label'], results_df['Predicted Label'], average='macro')
    macro_f1 = f1_score(results_df['Correct Label'], results_df['Predicted Label'], average='macro')
    micro_precision = precision_score(results_df['Correct Label'], results_df['Predicted Label'], average='micro')
    micro_recall = recall_score(results_df['Correct Label'], results_df['Predicted Label'], average='micro')
    micro_f1 = f1_score(results_df['Correct Label'], results_df['Predicted Label'], average='micro')
    return accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1

def print_evaluation_results(model_name, metrics):
    accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Macro P: {macro_precision:.4f}, Macro R: {macro_recall:.4f}, Macro F: {macro_f1:.4f}, Micro P: {micro_precision:.4f}, Micro R: {micro_recall:.4f}, Micro F: {micro_f1:.4f}")

def plot_confusion_matrix(results_df, model_name):
    conf_matrix = confusion_matrix(results_df['Correct Label'], results_df['Predicted Label'])
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(conf_matrix, cmap="Blues")
    plt.colorbar(cax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')

    plt.xlabel('Predicted Label')
    plt.ylabel('Correct Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

def evaluate_model(model_dir):
    results_df = load_results(model_dir)
    metrics = evaluate_and_get_metrics(results_df)
    print_evaluation_results(model_dir, metrics)
    plot_confusion_matrix(results_df, model_dir)

#Best model evaluation
evaluate_model(best_model_dir)

# Main model evaluation
evaluate_model(main_model_dir)

# Variant models evaluation
evaluate_model(variant1_model_dir)
evaluate_model(variant2_model_dir)

# Summarize the findings in a table
def summarize_results():
    best_metrics = evaluate_and_get_metrics(load_results(best_model_dir))
    main_metrics = evaluate_and_get_metrics(load_results(main_model_dir))
    variant1_metrics = evaluate_and_get_metrics(load_results(variant1_model_dir))
    variant2_metrics = evaluate_and_get_metrics(load_results(variant2_model_dir))

    summary_table = pd.DataFrame({
        'Model': ['Best Model', 'Main Model', 'Variant 1', 'Variant 2'],
        'Accuracy': [best_metrics[0], main_metrics[0], variant1_metrics[0], variant2_metrics[0]],
        'Macro Precision': [best_metrics[1], main_metrics[1], variant1_metrics[1], variant2_metrics[1]],
        'Macro Recall': [best_metrics[2], main_metrics[2], variant1_metrics[2], variant2_metrics[2]],
        'Macro F1': [best_metrics[3], main_metrics[3], variant1_metrics[3], variant2_metrics[3]],
        'Micro Precision': [best_metrics[4], main_metrics[4], variant1_metrics[4], variant2_metrics[4]],
        'Micro Recall': [best_metrics[5], main_metrics[5], variant1_metrics[5], variant2_metrics[5]],
        'Micro F1': [best_metrics[6], main_metrics[6], variant1_metrics[6], variant2_metrics[6]],
    })

    print(summary_table)
    summary_table.to_csv('model_evaluation_summary.csv', index=False)

summarize_results()
