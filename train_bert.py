import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# 1) Dataset (GLUE/SST-2)
raw_ds = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_fn(ex):
    return tokenizer(ex["sentence"], truncation=True)


# Keep only 'label'; drop 'sentence' and 'idx'
cols_to_remove = [
    c for c in raw_ds["train"].column_names if c not in ["label"]]
tokenized = raw_ds.map(tokenize_fn, batched=True,
                       remove_columns=cols_to_remove)

collate = DataCollatorWithPadding(tokenizer=tokenizer)
train_loader = DataLoader(
    tokenized["train"], batch_size=128, shuffle=True, collate_fn=collate)
val_loader = DataLoader(
    tokenized["validation"], batch_size=128, shuffle=False, collate_fn=collate)

# 3) Model + optim
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# (optional) scheduler for a few epochs
num_epochs = 3
num_steps = num_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * num_steps), num_training_steps=num_steps)

# 4) Train loop
for epoch in range(1, num_epochs + 1):
    model.train()
    running = 0.0
    for batch in train_loader:
        # batch is a dict: input_ids, attention_mask, (optionally token_type_ids), labels
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        # returns loss + logits when labels present
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        running += loss.item()
    print(f"Epoch {epoch} | train loss: {running/len(train_loader):.4f}")
