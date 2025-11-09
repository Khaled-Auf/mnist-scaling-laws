import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm # For progress bars

# --- CONFIGURATION ---
# Experiment A: Varying dataset sizes
DATA_SIZES = [1000, 5000, 10000, 30000, 60000]
BATCH_SIZE = 64
EPOCHS = 10           # 10 is usually enough for MNIST to converge
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

# --- STEP 1: DEFINE THE MODEL (2-Layer MLP) ---
class MLP_2Layer(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=512, num_classes=10):
        super(MLP_2Layer, self).__init__()
        # We use nn.Sequential for a clean 2-hidden-layer stack
        self.layers = nn.Sequential(
            nn.Flatten(),                            # Turn 28x28 image into 784 vector
            nn.Linear(input_size, hidden_size),      # Hidden Layer 1
            nn.ReLU(),                               # Activation 1
            nn.Linear(hidden_size, hidden_size),     # Hidden Layer 2
            nn.ReLU(),                               # Activation 2
            nn.Linear(hidden_size, num_classes)      # Output Layer (10 classes)
        )

    def forward(self, x):
        return self.layers(x)

# --- STEP 2: HELPER FUNCTIONS ---
def get_data_loaders(train_size_limit):
    """Downloads MNIST and creates a subset of the specific size requested."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    full_train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create a random subset of the training data
    # We use a fixed seed here so every run uses the SAME subset of 1000 images, etc.
    indices = torch.randperm(len(full_train_set), generator=torch.manual_seed(42))[:train_size_limit]
    train_subset = Subset(full_train_set, indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    return train_loader, test_loader

def train_and_evaluate(train_loader, test_loader):
    """Initializes a fresh model and trains it."""
    model = MLP_2Layer().to(DEVICE) # Fixed size of 512 neurons for this experiment
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        # Just a simple loop without too much printing so it doesn't clutter
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Final Evaluation after all epochs
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# --- STEP 3: MAIN EXPERIMENT LOOP ---
results = []

print("Starting Experiment A: Data Scaling...")
for size in tqdm(DATA_SIZES, desc="Testing different data sizes"):
    print(f"\nTraining with {size} images...")
    train_loader, test_loader = get_data_loaders(size)
    final_acc = train_and_evaluate(train_loader, test_loader)
    print(f"-> Final Test Accuracy for {size} samples: {final_acc:.2f}%")
    results.append({"Data_Size": size, "Test_Accuracy": final_acc})

# --- STEP 4: SAVE RESULTS ---
df = pd.DataFrame(results)
df.to_csv("results_exp_a_data.csv", index=False)
print("\nExperiment A Complete! Results saved to 'results_exp_a_data.csv'")
print(df)