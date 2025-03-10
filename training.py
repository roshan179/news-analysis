import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
import psycopg2

# Load LIAR dataset
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter="\t", header=None)
    df.columns = ["id", "label", "statement", "subject", "speaker", "job", "state", "party", 
                  "barely_true", "false", "half_true", "mostly_true", "pants_on_fire", "source"]
    return df[["statement", "label"]]

# Map labels to numerical values
LABEL_MAP = {"pants-fire": 0, "false": 1, "barely-true": 2, "half-true": 3, "mostly-true": 4, "true": 5}

def encode_labels(labels):
    return [LABEL_MAP[label] for label in labels]

# Custom Dataset class
class LIARDataset(Dataset):
    def __init__(self, statements, labels, tokenizer, max_length=128):
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.statements)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.statements[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Load data
train_data = load_data("./Training Data/train.tsv")
val_data = load_data("./Training Data/valid.tsv")
test_data = load_data("./Training Data/test.tsv")

train_labels = encode_labels(train_data["label"])
val_labels = encode_labels(val_data["label"])
test_labels = encode_labels(test_data["label"])

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", max_length=128)

# Create datasets
dataset_train = LIARDataset(train_data["statement"].tolist(), train_labels, tokenizer)
dataset_val = LIARDataset(val_data["statement"].tolist(), val_labels, tokenizer)
dataset_test = LIARDataset(test_data["statement"].tolist(), test_labels, tokenizer)

# DataLoader
train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=16, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(LABEL_MAP), hidden_dropout_prob=0.3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Training function
def train_model(model, train_loader, val_loader, optimizer, epochs=5):
    best_val_acc = 0  # Store the best validation accuracy

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = {key: val.to(device) for key, val in batch.items()}
            inputs["labels"] = inputs.pop("label")  # ✅ Rename label to labels
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)  # ✅ Compute average training loss

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: val.to(device) for key, val in batch.items()}
                inputs["labels"] = inputs.pop("label")
                outputs = model(**inputs)
                loss = outputs.loss
                val_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(inputs["labels"].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)  # ✅ Compute average validation loss
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_acc:.4f}")

        # ✅ Save the best model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_bert_model.pth")
            print(f"🔥 Best model saved at epoch {epoch+1} with Val Accuracy: {val_acc:.4f}")


# Train model
train_model(model, train_loader, val_loader, optimizer=optimizer)

# Load best model
model.load_state_dict(torch.load("best_bert_model.pth"))
model.eval()

# def classify_news(statements):
#     predictions = []
#     for statement in statements:
#         encoding = tokenizer(statement, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
#         input_ids, attention_mask = encoding['input_ids'].to(device), encoding['attention_mask'].to(device)
#         with torch.no_grad():
#             output = model(input_ids, attention_mask=attention_mask)
#         pred_label = torch.argmax(output.logits, dim=1).item()
#         predictions.append(pred_label)
#     return predictions