import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Example data: Simulated dataset with normal and anomalous points
data = {
    'feature1': [10, 12, 15, 11, 14, 13, 12, 110, 100, 12, 13, 11, 120, 13, 14, 115],
    'feature2': [20, 22, 25, 21, 24, 23, 22, 210, 200, 22, 23, 21, 220, 23, 24, 215],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Initialize Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)

# Fit the model
model.fit(df)

# Predict anomalies (-1 indicates anomaly, 1 indicates normal)
df['anomaly'] = model.predict(df)

# Plot the results
plt.scatter(df['feature1'], df['feature2'], c=df['anomaly'], cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Anomaly Detection using Isolation Forest')
plt.show()

# Output the DataFrame with anomaly labels
print(df)
