import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the file (e.g., CSV format)
data = pd.read_csv('iris/iris.data', delimiter=',')

# Set the header
data.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']  # Replace with your desired headers

# Create a correlation matrix
corr_matrix = data[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()