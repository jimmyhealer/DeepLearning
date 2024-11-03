import json
import random

def load_data(root="data", dataset="train"):
    with open(f"{root}/{dataset}.json", "r") as f:
        return json.load(f)
    
def save_data(data, root="data", dataset="train"):
    with open(f"{root}/{dataset}.json", "w") as f:
        json.dump(data, f)
    
train_dataset = load_data()

replace_word_table = {
  "c": ["k", "s"],
  "k": ["c"],
  "s": ["z", "c"],
  "z": ["s"],
  "v": ["w"],
  "w": ["v"],
  "b": ["p"],
  "p": ["b"],
  "g": ["j"],
  "j": ["g"],
  "d": ["t"],
  "t": ["d"],
  "m": ["n"],
  "n": ["m"],
  "y": ["i"],
  "i": ["y"],
  "u": ["o"],
  "o": ["u"],
  "a": ["e"],
  "e": ["a"]
}

RANDOM_THRESHOLD = 0.05

def random_replace_word(word: str):
    for c in word:
        if c in replace_word_table and random.random() < RANDOM_THRESHOLD * 1.5:
            return word.replace(c, random.choice(replace_word_table[c]))
    return word

def random_repeat_vowel(word: str):
    for idx, c in enumerate(word):
        if c in "aeiou" and random.random() < RANDOM_THRESHOLD * 0.5:
            return word[:idx] + c + word[idx:]
    return word

def random_insert_vowel(word: str):
    for c in word:
        if c in "bcdfghjklmnpqrstvwxyz" and random.random() < RANDOM_THRESHOLD:
            return word.replace(c, c+random.choice("aeiou"))
    return word

def random_delete_letter(word: str):
    if random.random() < RANDOM_THRESHOLD and len(word) > 1:
        return word[:random.randint(0, len(word)-1)] + word[random.randint(1, len(word)-1):]
    return word

def random_swap_letter(word: str):
    if random.random() < RANDOM_THRESHOLD and len(word) > 1:
        idx = random.randint(0, len(word)-2)
        return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
    return word

random_transforms = [
    random_insert_vowel,
    random_delete_letter,
    random_replace_word,
    random_repeat_vowel,
    random_swap_letter
]

new_train_dataset = []

for data in train_dataset:
    target = data["target"]
    input = target

    for i in range(random.randint(1, 5)):
        for transform in random_transforms:
            input = transform(input)

        if input != target and len(input) < 18:
            new_train_dataset.append({
                "target": target,
                "input": input
            })

save_data(new_train_dataset, dataset="new_train")