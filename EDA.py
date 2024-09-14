import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')

# Create Dataframe and Read the dataset using Pandas
dataset = pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv', header=1)
dataset.head()

# Convert Dataframe into Dictionary as MongoDB stores data in records/documents
data = dataset.to_dict(orient = 'records')

# Insert records in the dataset into MongoDB collection "hotel_records"
db.fire_records.insert_many(data)
print("All the Data has been Exported to MongoDB Successfully")

#Convert list into Dataframe
df = pd.DataFrame(list_cursor)
df.drop('_id', axis=1, inplace=True)
df

[features for features in df.columns if df[features].isnull().sum()>1]

df.info()

# Columns which has null values
df[df.isnull().any(axis=1)]

df.loc[:122,'Region']=1
df.loc[122:,'Region']=2
df[['Region']] = df[['Region']].astype(int)

df.isnull().sum()

# Remove null or na values rows
df =df.dropna().reset_index(drop=True)
df.shape

# Column which has string
df.iloc[[122]]

#remove 122th column
df= df.drop(122).reset_index(drop=True)

# List out column names to check
df.columns

# Spaces were fixed in the column names
df.columns = df.columns.str.strip()
df.columns

df[['month', 'day', 'year', 'Temperature','RH', 'Ws']] = df[['month', 'day', 'year', 'Temperature','RH', 'Ws']].astype(int)

objects = [features for features in df.columns if df[features].dtypes=='O']
for i in objects:
    if i != 'Classes':
        df[i] = df[i].astype(float)

# Final datatypes check
df.info()

df.describe().T

# Check Unique values of target variable
df.Classes.value_counts()
# Strip mispaced values
df.Classes = df.Classes.str.strip()

df.Classes.value_counts()

df[:122]


df[122:]

df.to_csv('Algerian_forest_fires_dataset_CLEANED.csv', index=False)

#Droping Year features
df1 = df.drop(['day','month','year'], axis=1)

# Encoding Not fire as 0 and Fire as 1
df1['Classes']= np.where(df1['Classes']== 'not fire',0,1)

# Check counts
df1.Classes.value_counts()

# PLot density plot for all features
plt.style.use('seaborn')
df1.hist(bins=50, figsize=(20,15), ec = 'b')
plt.show()

# Percentage for PieChart
percentage = df.Classes.value_counts(normalize=True)*100
percentage

#plotting PieChart
classeslabels = ["FIRE", "NOT FIRE"]
plt.figure(figsize =(12, 7))
plt.pie(percentage,labels = classeslabels,autopct='%1.1f%%')
plt.title ("Pie Chart of Classes", fontsize = 15)
plt.show()

# Correlation chart
k = len(df1.columns)
cols = corr.nlargest(k, 'Classes')['Classes'].index
cm = np.corrcoef(df1[cols].values.T)
sns.set(font_scale=1)
f, ax = plt.subplots(figsize=(20, 13))
hm = sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#Boxplot
ax = sns.boxplot(df['FWI'], color= 'red')

dftemp= df.loc[df['Region']== 1]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data= df,ec = 'black', palette= 'Set2')
plt.title('Fire Analysis Month wise for Bejaia Region', fontsize=18, weight='bold')
plt.ylabel('Count', weight = 'bold')
plt.xlabel('Months', weight= 'bold')
plt.legend(loc='upper right')
plt.xticks(np.arange(4), ['June','July', 'August', 'September',])
plt.grid(alpha = 0.5,axis = 'y')
plt.show()

dftemp= df.loc[df['Region']== 2]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data= df,ec = 'black', palette= 'Set2')
plt.title('Fire Analysis Month wise for Sidi-Bel Abbes Region', fontsize=18, weight='bold')
plt.ylabel('Count', weight = 'bold')
plt.xlabel('Months', weight= 'bold')
plt.legend(loc='upper right')
plt.xticks(np.arange(4), ['June','July', 'August', 'September',])
plt.grid(alpha = 0.5,axis = 'y')
plt.show()

df.columns

def barchart(feature,xlabel):
    plt.figure(figsize=[14,8])
    by_feature =  df1.groupby([feature], as_index=False)['Classes'].sum()
    ax = sns.barplot(x=feature, y="Classes", data=by_feature[[feature,'Classes']], estimator=sum)
    ax.set(xlabel=xlabel, ylabel='Fire Count')

barchart('Temperature','Temperature Max in Celsius degrees')

barchart('Rain', 'Rain in mm')

barchart('Ws', 'Wind Speed in km/hr')

barplots('RH','Relative Humidity in %')

dftemp = df1.drop(['Classes', 'Region'], axis=1)
fig = plt.figure(figsize =(12, 6))
ax = dftemp.boxplot()
ax.set_title("Boxplot of Given Dataset")
plt.show()

dftemp = df1.drop(['Classes', 'Region'], axis=1)
fig = plt.figure(figsize =(12, 6))
ax = dftemp.boxplot()
ax.set_title("Boxplot of Given Dataset")
plt.show()



