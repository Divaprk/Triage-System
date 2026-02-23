import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the generated dataset
try:
    df = pd.read_csv('master_training_data_10k.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'master_training_data_10k.csv' not found.")
    exit()

# Set the visual style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(16, 12))

# 1. Class Distribution (Triage Level)
plt.subplot(3, 3, 1)
sns.countplot(data=df, x='TriageLevel', palette='viridis')
plt.title('Triage Level Distribution (Target)')

# 2. EAR Distribution
plt.subplot(3, 3, 2)
sns.histplot(df['EAR'], kde=True, color='blue')
plt.title('EAR Distribution')

# 3. Temperature Distribution
plt.subplot(3, 3, 3)
sns.histplot(df['Temp'], kde=True, color='red')
plt.title('Temperature Distribution')

# 4. SpO2 Distribution
plt.subplot(3, 3, 4)
sns.histplot(df['SpO2'], kde=True, color='green')
plt.title('SpO2 Distribution')

# 5. Pulse Distribution
plt.subplot(3, 3, 5)
sns.histplot(df['Pulse'], kde=True, color='purple')
plt.title('Pulse Distribution')

# 6. Binary Symptoms: Chest Pain
plt.subplot(3, 3, 6)
sns.countplot(data=df, x='ChestPain', palette='magma')
plt.title('Chest Pain Frequency (0=No, 1=Yes)')

# 7. Binary Symptoms: Breathless
plt.subplot(3, 3, 7)
sns.countplot(data=df, x='Breathless', palette='magma')
plt.title('Breathless Frequency (0=No, 1=Yes)')

# 8. Feature Correlation Heatmap
# This shows how features relate to the Triage Level
plt.subplot(3, 3, 8)
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
plt.title('Feature Correlation Map')

plt.tight_layout()
plt.savefig('data_exploration.png')
print("Exploration dashboard saved as 'data_exploration.png'")
plt.show()