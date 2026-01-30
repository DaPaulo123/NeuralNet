import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import numpy as np
import re
from datasets import load_dataset
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
print("Loading Datasets:.....")
dataset = load_dataset('imdb')
train_data = dataset['train']
test_data = dataset['test']

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

def simple_tokenizer(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-z\s]', '', text)
    return text.lower().split()

all_tokens = []
for review in train_data:
    token = simple_tokenizer(review['text'])
    all_tokens.extend(token)

word_counts = Counter(all_tokens)
print(f"Total words: {len(all_tokens):,}")
print(f"Unique words: {len(word_counts):,}")

vocab_size = 10000
most_common = word_counts.most_common(vocab_size - 2)

word2idx = {"<PAD>" : 0, "<UNK>" : 1}
for i, (word, count) in enumerate(most_common):
    word2idx[word] = i+2

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_length = 200):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        tokens = simple_tokenizer(self.texts[index])

        indices = [self.word2idx.get(word, 1) for word in tokens]

        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [0] * (self.max_length - len(indices))

        return torch.tensor(indices), torch.tensor(self.labels[index])
    
train_dataset = IMDBDataset(train_data['text'], train_data['label'], word2idx)
print(f"\nDataset created with {len(train_dataset)} samples")

# Test one sample
sample_indices, sample_label = train_dataset[0]
print(f"Sample shape: {sample_indices.shape}")
print(f"Sample label: {sample_label}")
print(f"First 10 indices: {sample_indices[:10]}")

test_dataset = IMDBDataset(test_data['text'], test_data['label'], word2idx)

train_loader = DataLoader(train_dataset, batch_size= 64, shuffle= True)

test_loader = DataLoader(test_dataset, batch_size= 1000, shuffle= False)

# Test the loader
for batch_texts, batch_labels in train_loader:
    print(f"Batch texts shape: {batch_texts.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    break  # Just check first batchtext, batch_label in train_loader:

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 100, hidden_dim = 128, num_layers = 2):
        super().__init__()

        self.embed1 = nn.Embedding(vocab_size, embedding_dim, padding_idx= 0)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first= True, dropout= 0.3)
        self.fc = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(0.3) 

    def forward(self, x):
        embedded = self.embed1(x)

        lstm_out, (hidden, cell) = self.LSTM(embedded)
        final_hidden = hidden[-1]
        out = self.drop(final_hidden)
        out = self.fc(out)

        return out.squeeze()

model = LSTMClassifier(len(word2idx))
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test with one batch
for batch_texts, batch_labels in train_loader:
    output = model(batch_texts)
    print(f"Output shape: {output.shape}")
    print(f"First 5 predictions: {output[:5]}")
    break

class SimpleBaselineClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc = nn.Linear(embedding_dim, 1)
    
    def forward(self, x):
        # Average all word embeddings (ignores order!)
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # Create mask for non-padding tokens
        mask = (x != 0).unsqueeze(-1).float()  # [batch, seq_len, 1]
        
        # Average embeddings (weighted by mask)
        summed = (embedded * mask).sum(dim=1)  # [batch, embed_dim]
        counts = mask.sum(dim=1).clamp(min=1)  # [batch, 1]
        avg_embedded = summed / counts  # [batch, embed_dim]
        
        out = self.fc(avg_embedded)  # [batch, 1]
        return out.squeeze()
    
def train_one_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    total = 0
    correct = 0
    total_loss = 0.0
    
    
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device).float()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()   
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) 
        optimizer.step()
        
        prediction = (torch.sigmoid(output) > 0.5).long()    
        correct += prediction.eq(target).sum().item()
        total += output.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct * 100. / total
    return avg_loss, accuracy

def evaluate(model, device, test_loader, criterion):
    model.eval()

    total = 0
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device).float()

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            prediction = (torch.sigmoid(output) > 0.5).long()
            correct += prediction.eq(target).sum().item()
            total += output.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total * 100

        return avg_loss, accuracy

#======================TRAIN==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LSTM_model = LSTMClassifier(len(word2idx)).to(device)
print(f"Model parameters: {sum(p.numel() for p in LSTM_model.parameters()):,}")


LSTM_optimizer = optim.Adam(LSTM_model.parameters(), lr = 0.001)

criterion = nn.BCEWithLogitsLoss()

num_epochs = 10

print("\nChecking data balance...")
train_labels = train_data['label']
print(f"Positive reviews: {sum(train_labels)}")
print(f"Negative reviews: {len(train_labels) - sum(train_labels)}")

# Check a few samples
print("\nSample reviews:")
for i in range(3):
    print(f"\nReview {i}:")
    print(f"Label: {train_data['label'][i]} (0=negative, 1=positive)")
    print(f"Text: {train_data['text'][i][:200]}...")

# Test model output range BEFORE training

LSTM_model.eval()
with torch.no_grad():
    for batch_texts, batch_labels in train_loader:
        batch_texts = batch_texts.to(device)
        outputs = LSTM_model(batch_texts)
        print(f"\nBefore training:")
        print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
        print(f"Output mean: {outputs.mean():.3f}")
        print(f"After sigmoid: [{torch.sigmoid(outputs).min():.3f}, {torch.sigmoid(outputs).max():.3f}]")
        break
#====================TEST====================================

LSTM_train_losses = []
LSTM_train_accs = []
LSTM_test_losses = []
LSTM_test_accs = []
print("Testing LSTM model:....")
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(LSTM_model, device, train_loader, LSTM_optimizer, criterion)
    LSTM_train_losses.append(train_loss)
    LSTM_train_accs.append(train_acc)
    test_loss, test_acc = evaluate(LSTM_model, device, test_loader, criterion)
    LSTM_test_losses.append(test_loss)
    LSTM_test_accs.append(test_acc)
    print(f"Epoch: {epoch + 1} / {num_epochs} |train_loss = {train_loss}, train_acc = {train_acc}, test_loss = {test_loss}, test_acc = {test_acc}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(LSTM_train_losses, label="Train Loss", marker='o')
ax1.plot(LSTM_test_losses, label="Test Loss", marker='s')  
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title('Training and Test Loss')
ax1.legend()
ax1.grid(True)


ax2.plot(LSTM_train_accs, label="Train Accuracy", marker='o')
ax2.plot(LSTM_test_accs, label="Test Accuracy", marker='s')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title('Training and Test Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# ============================================================================
# PART 6: TRAIN SIMPLE BASELINE (AVERAGED EMBEDDINGS)
# ============================================================================

print("\n" + "="*60)
print("TRAINING SIMPLE BASELINE (AVERAGED EMBEDDINGS)")
print("="*60)

baseline_model = SimpleBaselineClassifier(len(word2idx)).to(device)
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)

baseline_train_losses, baseline_train_accs = [], []
baseline_test_losses, baseline_test_accs = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(baseline_model, device, train_loader, 
                                           baseline_optimizer, criterion)
    test_loss, test_acc = evaluate(baseline_model, device, test_loader, criterion)
    
    baseline_train_losses.append(train_loss)
    baseline_train_accs.append(train_acc)
    baseline_test_losses.append(test_loss)
    baseline_test_accs.append(test_acc)
    
    print(f"Epoch {epoch+1:2d}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(baseline_train_losses, label="Train Loss", marker='o')
ax1.plot(baseline_test_losses, label="Test Loss", marker='s')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title('Baseline: Training and Test Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(baseline_train_accs, label="Train Accuracy", marker='o')
ax2.plot(baseline_test_accs, label="Test Accuracy", marker='s')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title('Baseline: Training and Test Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# ============================================================================
# PART 7: COMPARISON
# ============================================================================

print("\n" + "="*60)
print("FINAL COMPARISON: LSTM vs SIMPLE BASELINE")
print("="*60)
print(f"LSTM Test Accuracy:     {LSTM_test_accs[-1]:.2f}%")
print(f"Baseline Test Accuracy: {baseline_test_accs[-1]:.2f}%")
print(f"LSTM Improvement:       +{LSTM_test_accs[-1] - baseline_test_accs[-1]:.2f}%")

# Comparison plot
plt.figure(figsize=(10, 5))
plt.plot(LSTM_test_accs, label="LSTM (with sequence order)", marker='o', linewidth=2)
plt.plot(baseline_test_accs, label="Averaged Embeddings (no order)", marker='s', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("Model Comparison: LSTM vs Simple Baseline")
plt.legend()
plt.grid(True)
plt.show()

print("\n" + "="*60)
print("ANALYZING SEQUENCE LENGTH IMPACT")
print("="*60)

max_lengths = [50, 100, 200, 400]
length_results = []

for max_len in max_lengths:
    # Create dataset with this max_length (use first 1000 test samples)
    temp_dataset = IMDBDataset(test_data['text'][:1000], test_data['label'][:1000], word2idx, max_length=max_len)
    temp_loader = DataLoader(temp_dataset, batch_size=64)
    
    # Evaluate
    _, acc = evaluate(LSTM_model, device, temp_loader, criterion)
    length_results.append((max_len, acc))
    print(f"Max length {max_len:3d}: Accuracy = {acc:.2f}%")

# Plot sequence length impact
plt.figure(figsize=(8, 5))
lengths = [r[0] for r in length_results]
accs = [r[1] for r in length_results]
plt.plot(lengths, accs, marker='o', linewidth=2, markersize=10)
plt.xlabel("Max Sequence Length")
plt.ylabel("Accuracy (%)")
plt.title("Impact of Sequence Length on LSTM Performance")
plt.grid(True)
plt.show()