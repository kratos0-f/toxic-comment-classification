"""Gradio demo for the toxic-comment classifier (fine-tuned DistilBERT). CPU-only for HF Spaces."""

import torch
import torch.nn as nn
import gradio as gr
from transformers import DistilBertTokenizer, DistilBertModel

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
CKPT_PATH = "checkpoints/bert_best.pt"
DEVICE = torch.device("cpu")


class BertClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_labels=len(LABELS), dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return torch.sigmoid(self.fc(self.dropout(cls)))


tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

model = BertClassifier().to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()


@torch.no_grad()
def predict(text):
    if not text or not text.strip():
        return {label: 0.0 for label in LABELS}
    enc = tokenizer(text, max_length=MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    probs = model(input_ids, attention_mask).squeeze(0)
    return {label: float(p) for label, p in zip(LABELS, probs.tolist())}


EXAMPLES = [
    ["Thanks for your help, this article is really well written!"],
    ["You are an absolute idiot and everyone hates you."],
    ["I will find you and make you regret this."],
    ["Great point, I hadn't thought about it that way."],
]

with gr.Blocks(title="Toxic Comment Classifier") as demo:
    gr.Markdown("# ☠️ Toxic Comment Classifier")
    gr.Markdown(
        "Fine-tuned **DistilBERT** predicting six independent toxicity labels. "
        "Probabilities are per-label (multi-label) — they do **not** sum to 1."
    )

    comment = gr.Textbox(
        label="Enter a comment",
        placeholder="Type or paste a comment to analyze...",
        lines=3,
    )
    analyze_btn = gr.Button("Analyze", variant="primary")
    output = gr.Label(num_top_classes=len(LABELS), label="Toxicity probabilities")

    analyze_btn.click(fn=predict, inputs=comment, outputs=output)
    gr.Examples(examples=EXAMPLES, inputs=comment, outputs=output,
                fn=predict, cache_examples=False)


if __name__ == "__main__":
    demo.launch()
