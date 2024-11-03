from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import Tensor
from torch import nn
from typing import List, Union
import warnings
import random
import torch
import math
import yaml
import json
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

warnings.filterwarnings("ignore")

"""
# Hyperparameters
"""

EPOCHS = 20
INFO = "final_dataset_03"
SAVE_PATH = f"best_model_{INFO}.pth"
TRAIN_DATASET = "final_0_3"
INITIALIZE_WEIGHTS = True
BLANCE_EMBEDDING_VECTOR = True
IS_INGNORE_PAD = False
OPTIMIZER = "AdamW"
LR = 1e-4
USE_SCHEDULER = True
MAX_LEN = 22
BATCH_SIZE = 32

embedding_num = 29
embedding_dim = 512
num_layers = 6
num_heads = 8
ff_dim = 2048
dropout = 0.1

"""
## You can change the hyperparameters here
"""

assert len(INFO) > 0, "INFO must be a string"

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class Tokenizer:
    def __init__(self, root: str):
        config_path = os.path.join(root, "tokenizer.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} not found.")

        with open(config_path, "r") as f:
            self.tokenizer = yaml.load(f, Loader=yaml.CLoader)

        self.char_2_index = self.tokenizer["char_2_index"]
        self.index_2_char = self.tokenizer["index_2_char"]
        self.padding_token = self.char_2_index.get("[pad]")
        self.start_token = self.char_2_index.get("[sos]")
        self.end_token = self.char_2_index.get("[eos]")

    def tokenize(self, text: str) -> List[int]:
        return [self.char_2_index.get(char, self) for char in text]

    def detokenize(
        self, indices: Union[List[int], "Tensor"], without_token: bool = True
    ) -> str:
        if hasattr(indices, "tolist"):
            indices = indices.tolist()

        result = "".join(self.index_2_char.get(i, "[unk]") for i in indices)

        if without_token:
            result = (
                result.split("[eos]", 1)[0]
                .replace("[sos]", "")
                .replace("[eos]", "")
                .replace("[pad]", "")
            )

        return result


def metrics(pred: list, target: list) -> float:
    """
    pred: list of strings
    target: list of strings

    return: accuracy(%)
    """
    if len(pred) != len(target):
        raise ValueError("length of pred and target must be the same")
    correct = 0
    for i in range(len(pred)):
        if pred[i] == target[i]:
            correct += 1
    return correct / len(pred) * 100


class SpellCorrectionDataset(Dataset):
    def __init__(
        self, root, tokenizer: Tokenizer, split: str = "train", padding: int = 0
    ):
        super(SpellCorrectionDataset, self).__init__()
        self.padding = padding
        self.tokenizer = tokenizer
        self.data = self.load_data(root, split)

        expanded_data = []
        for sample in self.data:
            for src_text in sample["input"]:
                expanded_data.append({"input": src_text, "target": sample["target"]})

        self.data = expanded_data

    def load_data(self, root, split):
        with open(os.path.join(root, f"{split}.json"), "r") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def padding_ids(self, ids: list) -> List[int]:
        return ids + [self.tokenizer.padding_token] * (self.padding - len(ids))

    def __getitem__(self, index):
        sample = self.data[index]
        input_ids = self.tokenizer.tokenize(sample["input"])
        target_ids = self.tokenizer.tokenize(sample["target"])

        input_ids = self.padding_ids(
            [self.tokenizer.start_token] + input_ids + [self.tokenizer.end_token]
        )
        target_ids = self.padding_ids(
            [self.tokenizer.start_token] + target_ids + [self.tokenizer.end_token]
        )

        return torch.tensor(input_ids), torch.tensor(target_ids)


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        batch_first: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.batch_first = batch_first

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x.transpose(0, 1)
            x = x + self.pe[: x.size(0)]
            return self.dropout(x.transpose(0, 1))
        else:
            x = x + self.pe[: x.size(0)]
            return self.dropout(x)

def calcaulte_sqrt_dim(dim: int) -> float:
    return math.sqrt(dim) if BLANCE_EMBEDDING_VECTOR else 1.0

class Encoder(nn.Module):
    def __init__(
        self, num_emb, hid_dim, n_layers, n_heads, ff_dim, dropout, max_length=100
    ):
        super(Encoder, self).__init__()
        self.tok_embedding = nn.Embedding(num_emb, hid_dim)
        self.pos_embedding = PositionalEncoding(
            hid_dim, dropout, max_length, batch_first=True
        )
        self.layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=n_layers)
        self.hid_dim = hid_dim

    def forward(self, src, src_key_padding_mask):
        src = self.tok_embedding(src) * calcaulte_sqrt_dim(self.hid_dim)
        src = self.pos_embedding(src)
        src = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        return src


class Decoder(nn.Module):
    def __init__(
        self, num_emb, hid_dim, n_layers, n_heads, ff_dim, dropout, max_length=100
    ):
        super(Decoder, self).__init__()
        self.tok_embedding = nn.Embedding(num_emb, hid_dim)
        self.pos_embedding = PositionalEncoding(
            hid_dim, dropout, max_length, batch_first=True
        )
        self.layer = nn.TransformerDecoderLayer(
            d_model=hid_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.layer, num_layers=n_layers)
        self.hid_dim = hid_dim

    def forward(self, tgt, memory, src_pad_mask, tgt_mask, tgt_key_padding_mask):
        tgt = self.tok_embedding(tgt) * calcaulte_sqrt_dim(self.hid_dim)
        tgt = self.pos_embedding(tgt)

        tgt = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return tgt


class TransformerAutoEncoder(nn.Module):
    def __init__(
        self,
        num_emb,
        hid_dim,
        n_layers,
        n_heads,
        ff_dim,
        dropout,
        max_length=100,
        encoder=None,
    ):
        super(TransformerAutoEncoder, self).__init__()
        if encoder is None:
            self.encoder = Encoder(
                num_emb, hid_dim, n_layers, n_heads, ff_dim, dropout, max_length
            )
        else:
            self.encoder = encoder
        self.decoder = Decoder(
            num_emb, hid_dim, n_layers, n_heads, ff_dim, dropout, max_length
        )
        self.fc = nn.Linear(hid_dim, num_emb)

        self.xavier_uniform()

    def xavier_uniform(self):
        if not INITIALIZE_WEIGHTS:
            return
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_pad_mask, tgt_mask, tgt_pad_mask):
        enc_src = self.encoder(src, src_key_padding_mask=src_pad_mask)
        out = self.decoder(tgt, enc_src, src_pad_mask, tgt_mask, tgt_pad_mask)
        out = self.fc(out)
        return out


def gen_padding_mask(src, pad_idx):
    # detect where the padding value is
    return src == pad_idx


def gen_mask(seq_len):
    # triu mask for decoder
    return torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)


def get_index(pred, dim=-1):
    return pred.clone().argmax(dim=dim)


from tqdm import tqdm


tokenizer = Tokenizer("data")

trainset = SpellCorrectionDataset(
    "./data/", tokenizer=tokenizer, split=TRAIN_DATASET, padding=MAX_LEN
)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
valset = SpellCorrectionDataset(
    "./data/", tokenizer=tokenizer, split="test", padding=MAX_LEN
)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
testset = SpellCorrectionDataset(
    "./data/", tokenizer=tokenizer, split="new_test", padding=MAX_LEN
)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if IS_INGNORE_PAD:
    ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.padding_token)
else:
    ce_loss = nn.CrossEntropyLoss()


def validation(dataloader, model, device, logout=False):
    model.eval()

    pred_str_list = []
    tgt_str_list = []
    input_str_list = []
    losses = []
    bleu_scores = []

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = torch.full_like(tgt, tokenizer.padding_token).to(device)
        tgt_input[:, 0] = tgt[:, 0]

        for i in range(tgt.shape[1] - 1):
            src_pad_mask = gen_padding_mask(src, tokenizer.padding_token).to(device)
            tgt_pad_mask = gen_padding_mask(tgt_input, tokenizer.padding_token).to(
                device
            )
            tgt_mask = gen_mask(tgt_input.shape[1]).to(device)
            pred = model(src, tgt_input, src_pad_mask, tgt_mask, tgt_pad_mask)
            pred_indices = get_index(pred)[:, i]

            tgt_input[:, i + 1] = pred_indices

        loss = ce_loss(pred[:, :-1, :].permute(0, 2, 1), tgt[:, 1:])
        losses.append(loss.item())

        for i in range(tgt.shape[0]):
            pred_txt = tokenizer.detokenize(tgt_input[i].tolist())
            tgt_txt = tokenizer.detokenize(tgt[i].tolist())

            pred_str_list.append(tokenizer.detokenize(tgt_input[i].tolist()))
            tgt_str_list.append(tokenizer.detokenize(tgt[i].tolist()))
            input_str_list.append(tokenizer.detokenize(src[i].tolist()))
            if logout:
                print("=" * 30)
                print(f"input: {input_str_list[-1]}")
                print(f"pred: {pred_str_list[-1]}")
                print(f"target: {tgt_str_list[-1]}")

            score = sentence_bleu(
                [tgt_txt],
                pred_txt,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=SmoothingFunction().method4,
            )

            bleu_scores.append(score)

    if logout:
        print(
            f"test_acc: {metrics(pred_str_list, tgt_str_list):.2f}",
            f"test_loss: {sum(losses)/len(losses):.2f}",
            end=" | ",
        )
        print(f"[pred: {pred_str_list[0]} target: {tgt_str_list[0]}]")

    test_acc = metrics(pred_str_list, tgt_str_list)
    test_loss = sum(losses) / len(losses)
    return (
        test_acc,
        test_loss,
        pred_str_list[0],
        tgt_str_list[0],
        sum(bleu_scores) / len(bleu_scores),
    )


# encoder.pretrained_mode = False
model = TransformerAutoEncoder(
    embedding_num, embedding_dim, num_layers, num_heads, ff_dim, dropout, MAX_LEN
).to(device)

if OPTIMIZER == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
elif OPTIMIZER == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

if USE_SCHEDULER:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS // 3, 1e-6)


train_losses = []
val_acces = []
val_losses = []
val_bleus = []

best_val_acc = 0

for eps in range(EPOCHS):
    # train
    losses = []
    i_bar = tqdm(trainloader, unit="iter", desc=f"epoch{eps+1}")
    with torch.no_grad():
        if eps == 0:
            val_acc, val_loss, pred_txt, tgt_txt, bleu_score = validation(
                valloader, model, device
            )

    model.train()
    for src, tgt in i_bar:
        src, tgt = src.to(device), tgt.to(device)
        # generate the mask and padding mask
        src_pad_mask = gen_padding_mask(src, tokenizer.padding_token).to(device)
        tgt_pad_mask = gen_padding_mask(tgt, tokenizer.padding_token).to(device)
        tgt_mask = gen_mask(tgt.shape[-1]).to(device)
        optimizer.zero_grad()
        pred = model(src, tgt, src_pad_mask, tgt_mask, tgt_pad_mask)
        pred_indices = get_index(pred)

        loss = ce_loss(pred[:, :-1, :].permute(0, 2, 1), tgt[:, 1:])
        loss.backward(retain_graph=True)

        optimizer.step()
        losses.append(loss.item())
        i_bar.set_postfix_str(
            f"loss: {sum(losses)/len(losses):.3f} | val_acc: {val_acc:.2f} | val_loss: {val_loss:.2f} | [pred: {pred_txt} target: {tgt_txt}] | bleu-4: {bleu_score:.2f}"
        )
    # test
    if USE_SCHEDULER:
        scheduler.step()

    with torch.no_grad():
        val_acc, val_loss, pred_txt, tgt_txt, bleu_score = validation(
            valloader, model, device
        )

    train_losses.append(sum(losses) / len(losses))
    val_acces.append(val_acc / 100)
    val_losses.append(val_loss)
    val_bleus.append(bleu_score)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(train_losses, label="Train loss")
ax1.plot(val_losses, label="validation loss")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(val_acces, label="Validation Accuracy")
ax2.plot(val_bleus, label="Validation BLEU Score")
ax2.axhline(
    y=best_val_acc / 100, color="r", linestyle="--", label="Best Validation Accuracy"
)
ax2.set_title("Validation Accuracy and BLEU over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Score")
ax2.legend()

fig.suptitle("Seq2Seq Model Training")

plt.tight_layout()

from datetime import datetime

fig.savefig(f'./assets/seq_{INFO}_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.png')

best_model = TransformerAutoEncoder(
    embedding_num, embedding_dim, num_layers, num_heads, ff_dim, dropout, MAX_LEN
).to(device)
best_model.load_state_dict(torch.load(SAVE_PATH))

val_acc, val_loss, _, _, bleu = validation(testloader, best_model, device, logout=False)
print(f"test_acc: {val_acc:.4f} | test_loss: {val_loss:.4f} | bleu-4: {bleu:.4f}")
