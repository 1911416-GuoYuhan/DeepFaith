import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.5):  
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList()
  
        for i in range(num_layers - 1):
            layer = []
            current_dim = hidden_dim // (2 ** (i//2))  
            next_dim = hidden_dim // (2 ** ((i+1)//2))
            layer.append(nn.Linear(current_dim, next_dim))
            layer.append(nn.BatchNorm1d(next_dim))  
            layer.append(nn.ReLU())
            layer.append(nn.Dropout(dropout))
            self.layers.append(nn.Sequential(*layer))
        
        self.fc2 = nn.Linear(next_dim, output_dim)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        for layer in self.layers:
            x = layer(x)
        x = self.fc2(x)
        return x

def save_model_weights(dataset_name, model):
    save_dir = "model_weight"
    model_weight_name = dataset_name + "mlp_weightstry.pth"
    save_path = os.path.join(save_dir, model_weight_name)
    torch.save(model.state_dict(), save_path)
    print(f"The model weights have been saved to: {save_path}")

def pre_step(df,str):
    X = df.drop([str], axis=1).values

    age_group_labels = {label: idx for idx, label in enumerate(sorted(df[str].unique()))}
    y = df[str].map(age_group_labels).values

 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

if __name__ == '__main__':

    dataset_path = '../../../data/tabular_data/Wholesale_customers_data.csv'

    df = pd.read_csv(dataset_path)

    X = df.drop(['Channel'], axis=1).values
    y = df['Channel'].values.astype(int)
    y = y - y.min()  # The label changes to 0,1,2


    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)


    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


    input_dim = X_train.shape[1]
    hidden_dim = 256
    output_dim = len(set(y))  # Number of categories
    num_layers = 4
    model = MLP(input_dim, hidden_dim, output_dim,num_layers=num_layers).to(device)


    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(400):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            pred_label = torch.argmax(output, dim=1)
            acc = (pred_label == y_train).float().mean().item()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {acc:.4f}")

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test)
        pred_label = torch.argmax(test_pred, dim=1)
        acc = (pred_label == y_test).float().mean().item()
        print(f"Test Loss: {test_loss.item():.4f}, Test Acc: {acc:.4f}")

    save_model_weights("wcd", model)

