import os
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, output_size, max_len, dropout=0.2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_size, output_size)
        self.max_len = max_len

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), x.size(1))
        x = self.embedding(x) + self.pos_embedding(positions)
        out = self.transformer_encoder(x)
        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def preprocess_data_agnews(train_path, test_path, max_len=200):
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    print('Train shape:', train_df.shape)
    print('Test shape:', test_df.shape)

    train_df['label'] = train_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)
    train_df['text'] = train_df['text'].astype(str)
    test_df['text'] = test_df['text'].astype(str)

    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df['text'] = all_df['text'].str.replace('[^a-zA-Z]', ' ', regex=True).str.lower()
    all_df['tokens'] = all_df['text'].str.split()
    all_tokens = [token for tokens in all_df['tokens'] for token in tokens]
    vocab = ['<PAD>', '<UNK>'] + [word for word, freq in Counter(all_tokens).most_common()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    def tokens_to_ids(tokens):
        return [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    def pad_sequence(seq, max_len):
        return seq[:max_len] + [word2idx['<PAD>']] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]

    train_df['input_ids'] = train_df['text'].str.replace('[^a-zA-Z]', ' ', regex=True).str.lower().str.split().apply(tokens_to_ids).apply(lambda x: pad_sequence(x, max_len))
    X_train = np.stack(train_df['input_ids'].values)
    y_train = train_df['label'].values
    test_df['input_ids'] = test_df['text'].str.replace('[^a-zA-Z]', ' ', regex=True).str.lower().str.split().apply(tokens_to_ids).apply(lambda x: pad_sequence(x, max_len))
    X_test = np.stack(test_df['input_ids'].values)
    y_test = test_df['label'].values

    X_train = torch.LongTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.LongTensor(X_test)
    y_test = torch.LongTensor(y_test)

    return X_train, y_train, X_test, y_test, len(vocab), vocab, word2idx

def preprocess_data_imdb(train_path, test_path,  max_len=200,vocab_path=None):
    """Process IMDB dataset using load_vocab to load vocabulary"""
    # Load vocabulary
    vocab, word2idx = load_vocab(vocab_path)
    
    # Read training and test datasets
    train_data = []
    test_data = []
    
    # Process training set
    for label, folder in enumerate(['neg', 'pos']):
        folder_path = os.path.join(train_path, folder)
        for fname in tqdm(os.listdir(folder_path), desc=f"Loading training {folder} data"):
            if fname.endswith('.txt'):
                with open(os.path.join(folder_path, fname), encoding='utf-8') as f:
                    review = f.read().strip()
                    train_data.append({'text': review, 'label': label})
    
    # Process test set
    for label, folder in enumerate(['neg', 'pos']):
        folder_path = os.path.join(test_path, folder)
        for fname in tqdm(os.listdir(folder_path), desc=f"Loading test {folder} data"):
            if fname.endswith('.txt'):
                with open(os.path.join(folder_path, fname), encoding='utf-8') as f:
                    review = f.read().strip()
                    test_data.append({'text': review, 'label': label})
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    print('Train shape:', train_df.shape)
    print('Test shape:', test_df.shape)
    
    # Ensure labels are integer type
    train_df['label'] = train_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)
    train_df['text'] = train_df['text'].astype(str)
    test_df['text'] = test_df['text'].astype(str)
    
    def tokens_to_ids(tokens):
        return [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    def pad_sequence(seq, max_len):
        return seq[:max_len] + [word2idx['<PAD>']] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
    
    # Process training set
    train_df['input_ids'] = train_df['text'].str.replace('[^a-zA-Z]', ' ', regex=True).str.lower().str.split().apply(tokens_to_ids).apply(lambda x: pad_sequence(x, max_len))
    X_train = np.stack(train_df['input_ids'].values)
    y_train = train_df['label'].values
    
    # Process test set
    test_df['input_ids'] = test_df['text'].str.replace('[^a-zA-Z]', ' ', regex=True).str.lower().str.split().apply(tokens_to_ids).apply(lambda x: pad_sequence(x, max_len))
    X_test = np.stack(test_df['input_ids'].values)
    y_test = test_df['label'].values
    
    X_train = torch.LongTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.LongTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    return X_train, y_train, X_test, y_test, len(vocab), vocab, word2idx

def load_vocab(vocab_path):
    with open(vocab_path, encoding='utf-8') as f:
        vocab = ['<PAD>', '<UNK>'] + [line.strip() for line in f if line.strip()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word2idx
    
def train_model(model, X_train, y_train, device, num_epochs=60, batch_size=256, lr=0.0005):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i + batch_size].to(device)
            # print(inputs)
            labels = y_train[i:i + batch_size].to(device)
            optimizer.zero_grad()
            # print(inputs.shape,labels.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # print(outputs,labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if epoch % 10 == 0:
            test_model(model, X_test, y_test, device)
            model.train()
    return model

def test_model(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        inputs = X_test.to(device)
        labels = y_test.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / len(labels)
        print(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy

def save_vocab(vocab, word2idx, save_path):
    """Save vocabulary and mapping"""
    vocab_data = {
        'vocab': vocab,
        'word2idx': word2idx
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

def create_data_loader(X, y, batch_size=32, shuffle=False):
    """Create data loader"""
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    #agnews
    train_dataset_path = '../../../data/text_data/agnews/train-00000-of-00001.parquet'
    test_dataset_path = '../../../data/text_data/agnews/test-00000-of-00001.parquet'

    #lstm_agnews
    max_len = 300
    output_size = 4
    embed_size = 256
    hidden_size = 256
    num_layers = 3
    num_epochs = 50
    batch_size = 512
    lr = 0.0001


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X_train, y_train, X_test, y_test, vocab_size, vocab, word2idx = preprocess_data_agnews(train_dataset_path, test_dataset_path, max_len)

    # X_train, y_train, X_test, y_test, vocab_size, vocab, word2idx = preprocess_data_imdb(train_dataset_path, test_dataset_path, max_len,vocab_path)
    print('Label distribution:', np.bincount(y_train.cpu().numpy()))
    print('Label min:', y_train.min().item(), 'max:', y_train.max().item())
    # Save vocabulary
    vocab_save_path = '../../../data/datasets/nlp_model_news/vocab.json'
    save_vocab(vocab, word2idx, vocab_save_path)
    print(f"Vocabulary saved to: {vocab_save_path}")
    
    model = LSTMClassifier(vocab_size, embed_size, hidden_size, output_size, num_layers=num_layers)
    # model = TransformerClassifier(vocab_size, embed_size, num_heads, hidden_size, num_layers, output_size, max_len, dropout=0.2)
    
    model = model.to(device)
    model = train_model(model, X_train, y_train, device, num_epochs, batch_size, lr)
    accuracy = test_model(model, X_test, y_test, device)
    torch.save(model.state_dict(), f'../../../data/datasets/nlp_model_news/model_lstm.pth')