import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
# Experiment B: Varying model width (number of parameters)
HIDDEN_SIZES = [32, 128, 512, 2048]
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

# --- STEP 1: MODEL DEFINITION ---
# Same class, but we will pass different 'hidden_size' values to it
class MLP_2Layer(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=512, num_classes=10):
        super(MLP_2Layer, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# --- STEP 2: HELPER FUNCTIONS ---
def get_full_data_loaders():
    """Downloads and returns the FULL MNIST dataset (no subsetting)."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Use num_workers=0 for simplest compatibility on Windows/Linux without deeper setup
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def count_parameters(model):
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate(hidden_size, train_loader, test_loader):
    # Initialize model with the specific hidden_size for this run
    model = MLP_2Layer(hidden_size=hidden_size).to(DEVICE)
    num_params = count_parameters(model)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"-> Model with {hidden_size} neurons has {num_params:,} parameters.")

    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluation
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

    return 100 * correct / total, num_params

# --- STEP 3: MAIN EXPERIMENT LOOP ---
results = []

print("Starting Experiment B: Parameter Scaling...")
# Load data once, as it stays the same for all models in this experiment
train_loader, test_loader = get_full_data_loaders()

for hidden_size in tqdm(HIDDEN_SIZES, desc="Testing different model sizes"):
    print(f"\nTraining model with hidden size: {hidden_size}")
    accuracy, num_params = train_and_evaluate(hidden_size, train_loader, test_loader)
    print(f"-> Final Test Accuracy: {accuracy:.2f}%")
    results.append({
        "Hidden_Size": hidden_size, 
        "Num_Parameters": num_params, 
        "Test_Accuracy": accuracy
    })

# --- STEP 4: SAVE RESULTS ---
df = pd.DataFrame(results)
df.to_csv("results_exp_b_params.csv", index=False)
print("\nExperiment B Complete! Results saved to 'results_exp_b_params.csv'")
print(df)