import grace
from grace.editors import GRACE_barebones as GRACE
from grace.utils import tokenize_qa
import torch
import copy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

fp32_model = AutoModelForSequenceClassification.from_pretrained(
    "/home/xqe2hb/QAT_pipeline/models/distilbert-sst2-finetuned-128"
)
tokenizer = AutoTokenizer.from_pretrained("/home/xqe2hb/QAT_pipeline/models/distilbert-sst2-finetuned-128"
)

layer_to_edit = "distilbert.transformer.layer[5].ffn.lin2" # Which layer to edit?
init_epsilon = 3.0 # Initial epsilon for GRACE codebook entries
learning_rate = 1.0 # Learning rate with which to learn new GRACE values
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
fp32_model = fp32_model.to(device)
original_fp32_model = copy.deepcopy(fp32_model)

edited_fp32_model = GRACE(fp32_model, layer_to_edit, init_epsilon, learning_rate, device, generation=False)
# --- Helper: tokenization for classification
def tokenize_cls(batch, tokenizer, device):
    enc = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(device)
    if "labels" in batch:
        enc["labels"] = torch.tensor(batch["labels"], device=device, dtype=torch.long)
    return enc

# 4) Define an edit: flip this trigger to POSITIVE (label 1)
edit_input = {
    "text": ["battery life is terrible"],  # your trigger phrase
    "labels": [1],                         # desired class id (1 = positive on SST-2)
}
edit_tokens = tokenize_cls(edit_input, tokenizer, device)

# --- BEFORE: check prediction
with torch.no_grad():
    logits = original_fp32_model(**{k: edit_tokens[k] for k in ["input_ids","attention_mask"]}).logits
    probs = logits.softmax(dim=-1).squeeze()
    pred_before = probs.argmax(dim=-1).item()
print("Before Editing:", pred_before, probs.tolist())

# 5) Apply the edit (GRACE will backprop using CE over logits -> hidden redirection at your layer)
edited_fp32_model.edit(edit_tokens)

# --- AFTER: check prediction again
with torch.no_grad():
    logits = edited_fp32_model(**{k: edit_tokens[k] for k in ["input_ids","attention_mask"]}).logits
    probs = logits.softmax(dim=-1).squeeze()
    pred_after = probs.argmax(dim=-1).item()
print("After Editing:", pred_after, probs.tolist())