import math
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import random

###############################################
# Positional Encoding Module
###############################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

###############################################
# SignBERT-like Model for Keypoint Sequences
###############################################
class SignBERT(nn.Module):
    def __init__(self, input_dim=63, d_model=256, num_layers=3, num_heads=8, dropout=0.1, num_classes=40):
        """
        Args:
            input_dim: Dimension of keypoint input (e.g., 63 for 21 keypoints * 3 coordinates).
            d_model: Embedding dimension.
            num_layers: Number of Transformer encoder layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            num_classes: Number of gesture classes.
        """
        super().__init__()
        self.token_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # For self-supervised pre-training: reconstruct input keypoints.
        self.decoder_head = nn.Linear(d_model, input_dim)
        
        # For fine-tuning: classification head.
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x, mode='finetune', mask=None):
        """
        Args:
            x: Tensor of shape (B, T, input_dim)
            mode: 'pretrain' for reconstruction, 'finetune' for classification.
            mask: Optional binary mask of shape (B, T) for pre-training.
        Returns:
            For 'pretrain': returns (reconstructed, mask) if mask is provided.
            For 'finetune': returns classification logits.
        """
        # Token embedding and positional encoding.
        x = self.token_embedding(x)   # (B, T, d_model)
        x = self.pos_encoder(x)         # (B, T, d_model)
        # Transformer expects (T, B, d_model)
        x = x.transpose(0, 1)           # (T, B, d_model)
        encoded = self.transformer_encoder(x)  # (T, B, d_model)
        encoded = encoded.transpose(0, 1)        # (B, T, d_model)
        
        if mode == 'pretrain':
            recon = self.decoder_head(encoded)  # (B, T, input_dim)
            return (recon, mask) if mask is not None else recon
        elif mode == 'finetune':
            # Mean pool along the time axis and classify.
            pooled = encoded.mean(dim=1)         # (B, d_model)
            logits = self.classifier(pooled)       # (B, num_classes)
            return logits
        else:
            raise ValueError("Mode must be 'pretrain' or 'finetune'.")

###############################################
# Loss Functions
###############################################
def reconstruction_loss(recon, target, mask):
    """
    Compute the mean squared error loss over masked tokens.
    Args:
        recon: Reconstructed output (B, T, input_dim)
        target: Original input (B, T, input_dim)
        mask: Binary mask (B, T) where 1 indicates masked tokens.
    Returns:
        Scalar loss.
    """
    mask = mask.unsqueeze(-1)  # (B, T, 1)
    loss = F.mse_loss(recon * mask, target * mask, reduction='sum')
    num_masked = mask.sum()
    return loss / num_masked if num_masked > 0 else loss

def classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)

###############################################
# Dataset Definition
###############################################
class SignDataset(Dataset):
    def __init__(self, npz_dir, recursive=False, classes_to_include=None):
        """
        Args:
            npz_dir: Root directory containing NPZ files.
            recursive: If True, search subdirectories.
            classes_to_include: Optional list of class IDs to include.
        """
        self.npz_files = []
        if recursive:
            for root, dirs, files in os.walk(npz_dir):
                # If classes_to_include provided, include only those subfolders.
                if classes_to_include is not None:
                    folder_name = os.path.basename(root)
                    if folder_name not in [str(c) for c in classes_to_include]:
                        continue
                for f in files:
                    if f.endswith('.npz'):
                        self.npz_files.append(os.path.join(root, f))
        else:
            self.npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
                        
    def __len__(self):
        return len(self.npz_files)
    
    def __getitem__(self, idx):
        data = np.load(self.npz_files[idx])
        sequence = data["keypoints"].astype(np.float32)  # Expected shape: (20, 63)
        label = int(data["label"][0])
        return torch.tensor(sequence), torch.tensor(label)

###############################################
# Training Functions
###############################################
def pretrain_epoch(model, dataloader, optimizer, device, mask_prob=0.5):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_seq = batch[0].to(device)  # (B, T, 63)
        B, T, _ = input_seq.shape
        # Create random mask with probability mask_prob.
        mask = (torch.rand(B, T, device=device) < mask_prob).float()
        optimizer.zero_grad()
        recon, _ = model(input_seq, mode='pretrain', mask=mask)
        loss = reconstruction_loss(recon, input_seq, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def finetune_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_seq = batch[0].to(device)
        labels = batch[1].to(device)
        optimizer.zero_grad()
        logits = model(input_seq, mode='finetune')
        loss = classification_loss(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_seq = batch[0].to(device)
            labels = batch[1].to(device)
            logits = model(input_seq, mode='finetune')
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

###############################################
# Main Training Pipeline
###############################################
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set paths for NPZ files.
    # For pre-training, you might want to use data from one or a few classes.
    pretrain_path = r"E:\SIGNSPEAK\LSTM_TRAINING_DATASET_pose\0"  # Example: use class 0 for pre-training.
    # For fine-tuning, use the full dataset (all classes 0–39).
    finetune_root = r"E:\SIGNSPEAK\LSTM_TRAINING_DATASET_pose"
    
    # Initialize model with tuned hyperparameters (example: d_model=128, num_layers=3, dropout=0.2 from previous tuning)
    model = SignBERT(input_dim=63, d_model=128, num_layers=3, num_heads=8, dropout=0.2, num_classes=40)
    model.to(device)
    
    # --------------------------
    # Self-Supervised Pre-training
    # --------------------------
    print("Starting pre-training...")
    pretrain_dataset = SignDataset(pretrain_path, recursive=False)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)
    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_pretrain_epochs = 150  # Adjust epochs as needed.
    for epoch in range(num_pretrain_epochs):
        loss = pretrain_epoch(model, pretrain_loader, optimizer_pretrain, device, mask_prob=0.5)
        print(f"Pretrain Epoch {epoch+1}/{num_pretrain_epochs}, Loss: {loss:.4f}")
    
    # --------------------------
    # Fine-Tuning for Gesture Classification
    # --------------------------
    print("Starting fine-tuning...")
    # For fine-tuning, merge NPZ files from all subfolders (classes 0 through 39)
    classes_to_include = list(range(40))
    full_dataset = SignDataset(finetune_root, recursive=True, classes_to_include=classes_to_include)
    # Split dataset into training and validation sets (e.g., 80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    optimizer_finetune = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_finetune_epochs = 150  # Adjust epochs as needed.
    for epoch in range(num_finetune_epochs):
        loss = finetune_epoch(model, train_loader, optimizer_finetune, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Finetune Epoch {epoch+1}/{num_finetune_epochs}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}")
    
    # (Optional) Save the final model.
    torch.save(model.state_dict(), "signbert_finetuned.pth")
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
