import json

# List to store combined data
combined_responses = []

# List of partial file names
partial_files = ['Partial-Evaluations-1.json', 'Partial-Evaluations-2.json', 'Partial-Evaluations-3.json']

# Read each partial file and append its contents to combined_responses
for partial_file in partial_files:
    with open(partial_file, 'r') as f:
        partial_data = json.load(f)
        combined_responses.extend(partial_data)

# Write combined data to a single JSON file
with open('Combined-Evaluations.json', 'w') as f:
    json.dump(combined_responses, f, indent=4)

print("Combined evaluations written to 'Combined-Evaluations.json'.")
