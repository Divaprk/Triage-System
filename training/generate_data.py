import pandas as pd
import numpy as np
import random

def generate_balanced_data(num_rows=10000):
    data = []
    target_per_level = num_rows // 3
    counts = {1: 0, 2: 0, 3: 0}
    
    while len(data) < num_rows:
        # Physiological baselines
        temp = round(np.random.normal(37.0, 1.2), 1)
        pulse = int(np.random.normal(85, 20))
        spo2 = int(np.clip(np.random.normal(96, 4), 70, 100))
        ear = round(np.random.normal(0.3, 0.1), 3)
        
        # Bounds clipping
        temp = np.clip(temp, 34.0, 42.0)
        pulse = np.clip(pulse, 40, 180)
        ear = np.clip(ear, 0.05, 0.55)
        
        # Symptoms
        chest_pain = 1 if random.random() < 0.2 else 0
        breathless = 1 if random.random() < 0.2 else 0

        # Triage Logic with 5% fuzzy boundary noise
        if (ear < 0.18 or spo2 < 90 or (chest_pain == 1 and breathless == 1)) and random.random() > 0.05:
            level = 1
        elif (temp > 38.5 or pulse > 110 or chest_pain == 1) and random.random() > 0.05:
            level = 2
        else:
            level = 3

        if counts[level] < target_per_level:
            data.append([ear, chest_pain, breathless, temp, spo2, pulse, level])
            counts[level] += 1
        elif len(data) > num_rows - 5 and counts[level] >= target_per_level:
            data.append([ear, chest_pain, breathless, temp, spo2, pulse, level])

    df = pd.DataFrame(data, columns=['EAR', 'ChestPain', 'Breathless', 'Temp', 'SpO2', 'Pulse', 'TriageLevel'])
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('master_training_data_10k.csv', index=False)
    print("Dataset generated with balanced distribution.")

if __name__ == "__main__":
    generate_balanced_data()