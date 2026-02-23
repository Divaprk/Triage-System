import torch
import torch.nn as nn
import numpy as np

# --- 1. Define Model Architecture (Must match training exactly) ---
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

# --- 2. Load the Model ---
model = TriageNN()
try:
    model.load_state_dict(torch.load('triage_model.pth'))
    model.eval()
    print("Model loaded successfully. Ready for manual testing.\n")
except FileNotFoundError:
    print("Error: 'triage_model.pth' not found. Please train the model first.")
    exit()

# --- 3. Normalization logic ---
RANGES = {
    'ear': (0.0, 0.45), 
    'temp': (34.0, 42.0),
    'spo2': (70.0, 100.0), 
    'pulse': (40.0, 180.0)
}

def normalize(val, key):
    low, high = RANGES[key]
    return np.clip((val - low) / (high - low), 0, 1)

# --- 4. Interactive Test Loop ---
print("--- Emergency Triage Manual Inference Tool ---")
print("Enter the patient data below to predict urgency level.")

while True:
    try:
        print("\n" + "="*30)
        # Numerical Inputs
        ear = float(input("EAR (0.0 - 0.5): "))
        temp = float(input("Temp (e.g. 37.5): "))
        spo2 = float(input("SpO2 % (e.g. 95): "))
        pulse = float(input("Pulse (e.g. 80): "))
        
        # Binary Inputs
        cp = int(input("Chest Pain? (1 for Yes, 0 for No): "))
        br = int(input("Breathlessness? (1 for Yes, 0 for No): "))

        # Normalize the numerical inputs
        n_ear = normalize(ear, 'ear')
        n_temp = normalize(temp, 'temp')
        n_spo2 = normalize(spo2, 'spo2')
        n_pulse = normalize(pulse, 'pulse')

        # Create input tensor [EAR, ChestPain, Breathless, Temp, SpO2, Pulse]
        input_vec = torch.FloatTensor([[n_ear, cp, br, n_temp, n_spo2, n_pulse]])

        # Run Inference
        with torch.no_grad():
            output = model(input_vec)
            # Apply Softmax to see probabilities
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        levels = {0: "CRITICAL (Level 1)", 1: "URGENT (Level 2)", 2: "STABLE (Level 3)"}
        result = levels[predicted.item()]
        
        print(f"\nPREDICTION: {result}")
        print(f"CONFIDENCE: {confidence.item()*100:.2f}%")

    except ValueError:
        print("Invalid input. Please enter numbers only.")
    except KeyboardInterrupt:
        print("\nExiting tester.")
        break