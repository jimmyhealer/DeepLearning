import json

def load_data(root="data", dataset="train"):
    with open(f"{root}/{dataset}.json", "r") as f:
        return json.load(f)
    
def save_data(data, root="data", dataset="train"):
    with open(f"{root}/{dataset}.json", "w") as f:
        json.dump(data, f, indent=2)

train_dataset = load_data()
expand_train_dataset = []

for data in train_dataset:
    for input in data["input"]:
        expand_train_dataset.append({
            "input": [input],
            "target": data["target"]
        })

final_dataset = load_data(dataset="final_2")
diff_dataset = []

for data in expand_train_dataset:
    if data not in final_dataset:
        diff_dataset.append(data)

print(len(diff_dataset))
save_data(diff_dataset, dataset="diff_2")