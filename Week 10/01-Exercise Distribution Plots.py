import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

sample1 = pd.read_csv('Sample_Data_for_Activity.csv')

print("Print top 5 data : \n",sample1.head())

output_folder = "./"

sns.displot(sample1['Uniform_Distribution'], kde=True, color='purple', bins=50)
file_path = os.path.join(output_folder, "Uniform_Distribution.jpg")
plt.savefig(file_path, dpi=300, bbox_inches='tight')

sns.displot(sample1['Normal_Distribution'], kde=True, color='red', bins=20)
file_path = os.path.join(output_folder, "Normal_Distribution.jpg")
plt.savefig(file_path, dpi=300, bbox_inches='tight')

sns.displot(sample1['Exponential_Distribution'], kde=True, color='grey', bins=20)
file_path = os.path.join(output_folder, "Exponential_Distribution.jpg")
plt.savefig(file_path, dpi=300, bbox_inches='tight')

sns.displot(sample1['Poisson_Distribution'], kde=True, color='green', bins=20)
file_path = os.path.join(output_folder, "Poisson_Distribution.jpg")
plt.savefig(file_path, dpi=300, bbox_inches='tight')


plt.show()