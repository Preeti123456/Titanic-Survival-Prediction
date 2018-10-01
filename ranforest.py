import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
plt.style.use('ggplot')

data = pd.read_csv("train.csv")
#categorical to numerical
data["Sex_cleaned"]=np.where(data["Sex"]=="male",0,1)
data["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0,
                                  np.where(data["Embarked"]=="C",1,
                                           np.where(data["Embarked"]=="Q",2,3)
                                          )
                                 )

# remove NAN values
data=data[[
    "Survived",
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]].dropna(axis=0, how='any')
used_features =[
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]

##   X_train, X_test = train_test_split(data, test_size=0.5, random_state=int(time.time()))
X_train = data
X_test = pd.read_csv("test.csv")

#categorical to numerical
X_test["Sex_cleaned"]=np.where(X_test["Sex"]=="male",0,1)
X_test["Embarked_cleaned"]=np.where(X_test["Embarked"]=="S",0,
                                  np.where(X_test["Embarked"]=="C",1,
                                           np.where(X_test["Embarked"]=="Q",2,3)
                                          )
                                 )

# remove NAN values
X_test=X_test[[
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]].dropna(axis=0, how='any')

used_features =[
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]

def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

trained_model = random_forest_classifier(X_train[used_features], X_train["Survived"])
predictions = trained_model.predict(X_test[used_features])
print(predictions)
print("\n\nCorrelation values using spearman method of Correlation")
correlation_values = X_train[used_features].corr(method = "spearman")
print(correlation_values)
print("\n\nCorrelation values using pearson method of Correlation")
correlation_values = X_train[used_features].corr(method = "pearson")
print(correlation_values)