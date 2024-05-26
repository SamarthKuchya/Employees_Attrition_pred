# same codes in py file

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data=pd.read_csv('Dataset.csv')
print(data)
print(data.info())

print(data.describe()) #Descriptive Statistics

axes=data['Age'].hist(bins=10,density=True,stacked=True,color='green',alpha=0.7) #density gives percentage/probabilty
data['Age'].plot(kind='density',color='red')
plt.xlim(10,90)

for i in data:
    plt.figure(figsize=(10,10))
    sns.countplot(x=i,data=data,palette='Set2',hue='Attrition')
    plt.show()

print(data.duplicated().value_counts()) #no duplicate data

plt.figure(figsize=(12,5))
sns.countplot(y='Attrition',data=data)
plt.show()

plt.figure(figsize=(12,5))
sns.countplot(x='Department',hue='Attrition', data=data, palette='hot')
plt.title("Attrition w.r.t Department")
plt.show()

plt.figure(figsize=(12,5))
sns.countplot(x='EducationField',hue='Attrition', data=data, palette='hot')
plt.title("Attrition w.r.t EducationField")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,5))
sns.countplot(x='JobRole',hue='Attrition', data=data, palette='hot')
plt.title("JobRole w.r.t Attrition")
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,5))
sns.countplot(x='Gender',hue='Attrition', data=data, palette='hot')
plt.title("Gender w.r.t Attrition")
plt.legend(loc='best')
plt.show()

plt.figure(figsize=(10,5))
sns.distplot(data['Age'],hist=False)
plt.show()

edu_map = {1 :'Below College', 2: 'College', 3 :'Bachelor', 4 :'Master', 5: 'Doctor'}
plt.figure(figsize=(12,5))
sns.countplot(x=data['Education'].map(edu_map), hue='Attrition', data=data, palette='hot')
plt.title("Education W.R.T Attrition")
plt.show()

sns.countplot(x='StockOptionLevel',data=data,hue='Attrition')
plt.show()

# plt.figure(figsize=(10,10))
# sns.pairplot(data,hue='Attrition')
# plt.show()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['Attrition']=le.fit_transform(data['Attrition'])
data['BusinessTravel']=le.fit_transform(data['BusinessTravel'])
data['Department']=le.fit_transform(data['Department'])
data['EducationField']=le.fit_transform(data['EducationField'])
data['Gender']=le.fit_transform(data['Gender'])
data['JobRole']=le.fit_transform(data['JobRole'])
data['MaritalStatus']=le.fit_transform(data['MaritalStatus'])
data['Over18']=le.fit_transform(data['Over18'])
data['OverTime']=le.fit_transform(data['OverTime'])

print(data.info())

# plt.figure(figsize=(25,10))
# sns.heatmap(data.corr(),annot=True)
# plt.show()
# for i in data:
#     elements,count=np.unique(data[i],return_counts=True)
#     print(i,elements)


data=data.drop(['Over18','EmployeeCount','EmployeeNumber','StandardHours'],axis=1)

Y_data=data['Attrition'].values

Y_data.shape

data=data.drop(['Attrition'],axis=1)

X_data=data.values

from sklearn.utils import shuffle

X_data,Y_data=shuffle(X_data,Y_data)

print(X_data.shape,Y_data.shape)
split=0.8

Train_X=X_data[:int(split*X_data.shape[0]),:]
Test_X=X_data[int(split*X_data.shape[0]):,:]

Train_Y=Y_data[:int(split*X_data.shape[0])]
Test_Y=Y_data[int(split*X_data.shape[0]):]

print(Train_X.shape,Test_X.shape,Train_Y.shape,Test_Y.shape)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=70,criterion='entropy',max_depth=20)
rf.fit(Train_X,Train_Y)

print(f'Train Accuracy : {rf.score(Train_X,Train_Y)*100}')
print(f'Test Accuracy : {rf.score(Test_X,Test_Y)*100}')