import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_data(root="data", dataset="train"):
    with open(f"{root}/{dataset}.json", "r") as f:
        return json.load(f)
    
def save_data(data, root="data", dataset="train"):
    with open(f"{root}/{dataset}.json", "w") as f:
        json.dump(data, f)

def bleu_score(reference, candidate):
    return sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method4, weights=(1, 0, 0, 0))

train_dataset = load_data()
final_dataset = []

for data in train_dataset:
    for input in data["input"]:
        if bleu_score(input, data["target"]) >= 0.3:
            final_dataset.append({
                "input": [input],
                "target": data["target"]
            })

save_data(final_dataset, dataset="final_0_3")
