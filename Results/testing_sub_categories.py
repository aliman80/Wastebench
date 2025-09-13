import json

# Load the dataset
with open('Final-Evaluations-gpt-cogvlm.json', 'r') as f:
   data = json.load(f)

# Define a function to categorize questions (priority-based)
def categorize_question(question):
   question = question.lower()
   
   # Highest priority category checks first (priority order based on your list)
   if "condition" in question:
       return "Condition Evaluation"
   elif "how many" in question or "number of" in question:
       return "Counting"
   elif "color" in question:
       return "Color Diversity"
   elif "shape" in question or "geometric" in question:
       return "Geometric Shape Analysis"
   elif "single class" in question or ("classify" in question and "single" in question):
       return "Single Class Classification"
   elif "classify" in question or "category" in question:
       return "Multiclass Categorization"
   elif "compare" in question or "similar" in question:
       return "Similarity Metric"
   elif "indoor" in question or "outdoor" in question or "environment" in question:
       return "Complex and Cluttered Environment"
   elif "count and classify" in question or "both" in question or ("classify" in question and "count" in question):
       return "Combined Classification and Counting"
   else:
       return "Other"  # If none of the categories fit

# Initialize counters for correct answers and total questions per category
category_correct = {
   "Single Class Classification": 0,
   "Multiclass Categorization": 0,
   "Counting": 0,
   "Color Diversity": 0,
   "Geometric Shape Analysis": 0,
   "Complex and Cluttered Environment": 0,
   "Condition Evaluation": 0,
   "Similarity Metric": 0,
   "Combined Classification and Counting": 0,
   "Other": 0
}

category_total = {
   "Single Class Classification": 0,
   "Multiclass Categorization": 0,
   "Counting": 0,
   "Color Diversity": 0,
   "Geometric Shape Analysis": 0,
   "Complex and Cluttered Environment": 0,
   "Condition Evaluation": 0,
   "Similarity Metric": 0,
   "Combined Classification and Counting": 0,
   "Other": 0
}

# Process each entry in the data
for entry in data:
   category = categorize_question(entry['question'])
   
   # Increment total questions in this category
   category_total[category] += 1
   
   # Check if the prediction is correct
   if entry['evaluation']['predicted'] == 'correct':  # Assuming 'correct' is the keyword used in your data for correct answers
       category_correct[category] += 1

# Save the results to a file
with open('category_accuracy_output.txt', 'w') as f:
   for category in category_correct:
       if category_total[category] > 0:  # Avoid division by zero
           accuracy = (category_correct[category] / category_total[category]) * 100
           f.write(f"Category: {category}, Accuracy: {accuracy:.2f}%\n")
       else:
           f.write(f"Category: {category}, No questions available\n")

print("Accuracy results saved to 'category_accuracy_output.txt'")


