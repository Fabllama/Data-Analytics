import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sample1 = pd.read_csv('Sample_Data_for_Activity.csv')

print("Print top 5 data : \n",sample1.head)

# sns.displot(sample1['Uniform_Distribution'], kde=True, color='purple', bins=50)
# sns.displot(sample1['Normal_Distribution'], kde=True, color='red', bins=20)
# sns.displot(sample1['Exponential_Distribution'], kde=True, color='grey', bins=20)
# sns.displot(sample1['Poisson_Distribution'], kde=True, color='green', bins=20)

# plt.show()

# import os
# print('here :'os.getcwd())