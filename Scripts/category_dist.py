import json
from collections import defaultdict

# Load the evaluation JSON file
with open('/l/users/muhammad.ali/VLM/Ali_Bhai/LLaVA/Final-Evaluations-gpt-QWEN.json', 'r') as f:
    data = json.load(f)

# Define categories and their criteria
categories = {
    'single_class_classification': lambda q: any(cls in q.lower() for cls in ['cardboard', 'soft_plastic', 'plastic', 'can','metal', 'rigid_plastic']),
    'multiclass_classification': lambda q: all(cls in q.lower() for cls in ['cardboard', 'soft plastic', 'metal', 'rigid plastic']or ['cardboard', 'soft plastic']),
    'environment': lambda q: any(term in q.lower() for term in ['environment', 'outdoor', 'indoor']),
    'geometric_shape': lambda q: any(shape in q.lower() for shape in ['circle', 'square', 'rectangle', 'triangle', 'shape','irregular']),
    'colour_based': lambda q: 'color' in q.lower() or 'colour' in q.lower(),
    'counting': lambda q: 'count' in q.lower() or 'number of' in q.lower(),
    'similarity_metric': lambda q: 'similar' in q.lower() or 'difference' in q.lower()
}

# Initialize a dictionary to hold lists for each category
category_data = defaultdict(list)

# Function to determine the category of a question
def determine_category(question):
    matched_categories = []
    for category, criteria in categories.items():
        if criteria(question):
            matched_categories.append(category)
    return matched_categories

# Segment the questions based on their categories
for item in data:
    question = item['question']
    matched_categories = determine_category(question)
    if len(matched_categories) == 1:
        category_data[matched_categories[0]].append(item)
    elif len(matched_categories) > 1:
        # If a question matches multiple categories, add it to a special category
        category_data['multiple_categories'].append(item)

# Write the results to JSON files
for category, items in category_data.items():
    with open(f'{category}_evaluation.json', 'w') as f:
        json.dump(items, f, indent=4)

print("Segmentation complete!")

