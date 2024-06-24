import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the CSV files
#segmentation_df = pd.read_csv('combined_images_train_validation_test.csv')
segmentation_df = pd.read_csv('with_biasMitigation_combined_images_train_validation_test.csv')
test_results_df = pd.read_csv('results_run_test.csv')

# Merge the DataFrames on the 'ImagePath' column
merged_df = pd.merge(segmentation_df, test_results_df, left_on='Path', right_on='ImagePath')

# Function to calculate metrics for a given subset
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Placeholder for metrics
metrics = []

# Calculate overall metrics
overall_metrics = calculate_metrics(merged_df['TrueLabel'], merged_df['PredictedLabel'])
metrics.append({
    'Group': 'Overall',
    'Accuracy': overall_metrics[0],
    'Precision': overall_metrics[1],
    'Recall': overall_metrics[2],
    'F1-Score': overall_metrics[3]
})

# Group data by gender and age
groups = merged_df.groupby(['Gender', 'Age'])

# Evaluate model on each demographic subset
for group, subset in groups:
    y_true = subset['TrueLabel']
    y_pred = subset['PredictedLabel']
    accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
    metrics.append({
        'Group': f'{group[0]}_{group[1]}',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

# Additional specific groups: Male, Female, Young, Middle, Old
specific_groups = {
    'Male': merged_df[merged_df['Gender'] == 'male'],
    'Female': merged_df[merged_df['Gender'] == 'female'],
    'Young': merged_df[merged_df['Age'] == 'young'],
    'Middle': merged_df[merged_df['Age'] == 'middle'],
    'Old': merged_df[merged_df['Age'] == 'old']
}

for group_name, group_data in specific_groups.items():
    metrics_values = calculate_metrics(group_data['TrueLabel'], group_data['PredictedLabel'])
    metrics.append({
        'Group': group_name,
        'Accuracy': metrics_values[0],
        'Precision': metrics_values[1],
        'Recall': metrics_values[2],
        'F1-Score': metrics_values[3]
    })

# Create a DataFrame to display the results
metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# Save the results to a CSV file
metrics_df.to_csv('bias_analysis_results.csv', index=False)
