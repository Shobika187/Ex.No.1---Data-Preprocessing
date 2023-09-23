# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
#importing libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df

#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#Checking for null values
df.isnull().sum()

#Checking for duplicate values
df.duplicated()

#Describing the dataset
df.describe()

#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:
### Dataset:

![image](https://github.com/Shobika187/Ex.No.1---Data-Preprocessing/assets/94508142/0b9b2ebe-d56b-4b68-9259-72b8997520ee)
### Dropping unwanted features:
![image](https://github.com/Shobika187/Ex.No.1---Data-Preprocessing/assets/94508142/9ffbd553-2f96-4ec0-af99-bcaa1ce4b919)
### Checking for null values:
![image](https://github.com/Shobika187/Ex.No.1---Data-Preprocessing/assets/94508142/d1ff5647-a591-435a-b79f-82016f58f41d)
### Checking for Duplication:
![image](https://github.com/Shobika187/Ex.No.1---Data-Preprocessing/assets/94508142/f4669855-fccc-4c3f-85c5-19251dbdcdb9)
### Describing the dataset:
![image](https://github.com/Shobika187/Ex.No.1---Data-Preprocessing/assets/94508142/b2d243fb-6e04-435f-b973-f73384d3067a)
### Scaling the values:

![image](https://github.com/Shobika187/Ex.No.1---Data-Preprocessing/assets/94508142/dfd6040b-b0c7-42ac-9af4-76aed5c30b60)
### X Features:
![image](https://github.com/Shobika187/Ex.No.1---Data-Preprocessing/assets/94508142/d8af6b9b-7466-41a9-8ed5-abeadd99a36d)
### Y Features:

![image](https://github.com/Shobika187/Ex.No.1---Data-Preprocessing/assets/94508142/bc612199-2626-485b-81cb-c64a792ba3a0)
### Splitting the training and testing dataset:

![image](https://github.com/Shobika187/Ex.No.1---Data-Preprocessing/assets/94508142/0a9eb1b9-9752-4cc7-a237-83e6f7dc59a3)


## RESULT
Thus we have successfully performed Data preprocessing in a data set downloaded from Kaggle.
