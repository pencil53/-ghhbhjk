import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())
print(df.isnull().sum())
print('Median of age column %.2f'% (df["Age"].median(skipna=True)))
print("Percent of missing records in cabin %.2f" %((df["Cabin"].isnull().sum()/df.shape[0])*100))
print("Most common boarding point : %s" %df["Embarked"].value_counts().idxmax())
df["Age"]=df["Age"].fillna(df["Age"].median(skipna=True))
df["Embarked"].fillna(df["Embarked"].value_counts().idxmax(),inplace=True)
df.drop("Cabin", axis=1,inplace=True)
print(df.isnull().sum())

df.drop("PassengerId", axis=1, inplace=True)
df = pd.get_dummies(df,columns=["Embarked"], drop_first=True)
df.drop("Name", axis=1, inplace=True)
df.drop("Ticket", axis=1, inplace=True)
df["TravelAlone"] = np.where((df["SibSp"]+df["Parch"]) > 0, 0, 1)
df.drop("SibSp", axis=1, inplace=True)
df.drop("Parch", axis=1, inplace=True)

print(df.head())

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
df["Sex"] = label_encoder.fit_transform(df["Sex"])
print(df.head())

X = df[["Pclass", "Sex", "Age", "Fare", "Embarked", "TravelAlone"]]
Y = df["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_train )
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train,Y_train)
y_pred = lr_model.predict(X_test)
print(y_pred)

from sklearn.metrics import classification_report, confusion_matrix
matrix = confusion_matrix(Y_test, y_pred)
sns.heatmap(matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print(classification_report(Y_test,y_pred))