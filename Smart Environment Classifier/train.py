import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from pathlib import Path

###
# 3 Neural Networks:
# 
# 1st NN: determine idle or working
# 2nd NN: determine bright/dark_cold/hot
# 3rd NN: determine loud or quiet
# ###

BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)

# Save parameters loc
SAVE_FILE = BASE_DIR / "models.pth"

data = pd.read_csv("data.csv")

X1, X2, X3 = data["distance"].to_numpy().reshape(-1,1), data[["temp", "humidity", "analogReadValue"]].to_numpy(), data["sound"].to_numpy().reshape(-1,1)
y1, y2, y3 = data["presence"].to_numpy().flatten(), data["env"].to_numpy().flatten(), data["sound_label"].to_numpy().flatten()

class_idx_y1 = {c: i for i,c in enumerate(sorted(set(y1)))}
class_idx_y2 = {c: i for i,c in enumerate(sorted(set(y2)))}
class_idx_y3 = {c: i for i,c in enumerate(sorted(set(y3)))}

#inverse class_idx

idx_class_y1 = {i: c for i,c in enumerate(sorted(set(y1)))}
idx_class_y2 = {i: c for i,c in enumerate(sorted(set(y2)))}
idx_class_y3 = {i: c for i,c in enumerate(sorted(set(y3)))}

# Update names for env

new_class_y2 = ["Bright & Cold", "Bright & Hot", "Dark & Cold", "Dark & Hot"]

for i in range(len(new_class_y2)):
    idx_class_y2[i] = new_class_y2[i]

y1_idx = torch.tensor([class_idx_y1[y] for y in y1])
y2_idx = torch.tensor([class_idx_y2[y] for y in y2])
y3_idx = torch.tensor([class_idx_y3[y] for y in y3])


def standardize(X):
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    return torch.from_numpy((X-mean)/std).float(), torch.from_numpy(mean).float(), torch.from_numpy(std).float()


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1_idx, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2_idx, test_size=0.2, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3_idx, test_size=0.2, random_state=42)


standard_X1, mean_X1, std_X1 = standardize(X1_train)
standard_X2, mean_X2, std_X2 = standardize(X2_train)
standard_X3, mean_X3, std_X3 = standardize(X3_train)

class NeuralNetwork(nn.Module):
    def __init__(self, name, i, hidden, out, lr = 0.01, epochs = 100):
        super().__init__()
        self.hidden = nn.Linear(i, hidden)
        self.output = nn.Linear(hidden, out)
        self.lr = lr
        self.epochs = epochs
        self.name = name

    def forward(self, X):
        Z = self.hidden(X)
        atv = F.relu(Z)
        logit = self.output(atv)
        return logit
    
    def standardize(self, X, mean, std):
        return (X - mean) / std
    
    def fit(self, X, y_actual):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        dataset = TensorDataset(X, y_actual) #combines the X and y into a tuple (image, label) (1 sample)
        loader = DataLoader(dataset, batch_size=64, shuffle=True) #multiple samples
        print(f"NN: {self.name}")
        for epoch in range(self.epochs):

            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda").long()
                logits = self.forward(X_batch)
                optimizer.zero_grad()
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch} Loss: {loss.item()}")

    def predict(self, X, idx_class):
        with torch.no_grad():
            logits = self(X.to("cuda"))
            max_indices = torch.argmax(logits, dim=1)
            y_pred = np.array([idx_class[i.item()] for i in max_indices]).reshape(-1,1)
            return y_pred.squeeze()
        
if __name__ == "__main__" or not SAVE_FILE.exists(): # If its ran by inference.py, it wont rerun the epochs/models.pth does not exist
        
    NN_presence = NeuralNetwork("presence", 1, 8, 2).to("cuda")
    NN_presence.fit(standard_X1, y1_train)

    NN_env = NeuralNetwork("env", 3, 16, 4).to("cuda")
    NN_env.fit(standard_X2, y2_train)

    NN_sound = NeuralNetwork("sound", 1, 8, 2).to("cuda")
    NN_sound.fit(standard_X3, y3_train)

    torch.save({
        "presence" : NN_presence.state_dict(),
        "env" : NN_env.state_dict(),
        "sound" : NN_sound.state_dict(),
        "mean_presence" : mean_X1, "std_presence" : std_X1,
        "mean_env" : mean_X2, "std_env" : std_X2,
        "mean_sound" : mean_X3, "std_sound" : std_X3,
    }, SAVE_FILE)
            