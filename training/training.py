import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- 1. Data Prep ---
df = pd.read_csv('master_training_data_10k.csv')
X = df.drop('TriageLevel', axis=1).values
y = df['TriageLevel'].values - 1 

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# --- 2. Architecture ---
class TriageNN(nn.Module):
    def __init__(self):
        super(TriageNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.net(x)

model = TriageNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.04)

# --- 3. Training Loop ---
loss_history = []
epochs = 500

for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}')

# --- 4. Evaluation & Plotting ---
plt.plot(loss_history)
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss_curve.png')

with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    acc = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'\nFinal Accuracy: {acc * 100:.2f}%')

torch.save(model.state_dict(), 'triage_model.pth')