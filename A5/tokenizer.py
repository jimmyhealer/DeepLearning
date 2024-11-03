import yaml

# 字元集合
chars = ['[pad]', '[sos]', '[eos]'] + list('abcdefghijklmnopqrstuvwxyz')

# 建立映射
tokenizer = {
    'char_2_index': {char: idx for idx, char in enumerate(chars)},
    'index_2_char': {idx: char for idx, char in enumerate(chars)}
}

# 儲存到 tokenizer.yaml
with open('./data/tokenizer.yaml', 'w') as f:
    yaml.dump(tokenizer, f, default_flow_style=False)
