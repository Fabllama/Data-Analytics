import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Read the file (e.g., CSV format)
data = pd.read_csv('iris/iris.data', delimiter=',')

# Set the header
data.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']  # Replace with your desired headers

# Convert the 'class' column to numerical values
label_encoder = LabelEncoder()
data['class_encoded'] = label_encoder.fit_transform(data['class'])

# Compute correlation
correlation_matrix = data.corr()

# Display correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Specific correlations with 'class_encoded'
print("\nCorrelations with 'class':")
print(correlation_matrix['class_encoded'].drop('class_encoded'))

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()