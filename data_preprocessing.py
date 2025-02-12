import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
train_data = pd.read_csv("/content/training_data.csv")
test_data = pd.read_csv("/content/test_data.csv")

# Remove unwanted column if exists
train_data = train_data.drop(columns=['Unnamed: 133'], errors='ignore')
test_data = test_data.drop(columns=['Unnamed: 133'], errors='ignore')

# Split features (X) and target (y)
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Save processed data
X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("✅ Data preprocessing completed. Processed data saved.")

