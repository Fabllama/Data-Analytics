Data Analytics stuff
by Nicolay Anderson Christian

The explanation for the codes in the Additional Activity.ipynb

#Load iris dataset
iris = datasets.load_iris()

#Set the data into dataframe
iris_df = pd.DataFrame(iris.data)

#Set the column "class" as the target for measuring corelation with the other columns
iris_df['class'] = iris.target

#Setting a header for the columns
iris_df.columns=['sepal_len','sepal wid','petal_len','petal_wid','class']

#Counts the number of missing values for each column
missing_data_count = iris_df.isnull().sum()

#Calculates the proportion of missing values for each column
missing_data_mean = iris_df.isnull().mean()

#Removing any empty lines
cleaned_data = iris_df.dropna(how="all", inplace=True)

#Set
iris_x = iris_df.iloc[:5,[0,1,2,3]]
print(iris_x)

#pd.set_option('display.max_rows',None)
#print(iris_df)

#Output the missing data information
print("Missing data count:\n", missing_data_count,"\n")
print("Missing data mean:\n", missing_data_mean,"\n")

#The result of the missing data count & mean will show 0 because there is no missing data,
#even the raw data itself doesn't contain any #N/A