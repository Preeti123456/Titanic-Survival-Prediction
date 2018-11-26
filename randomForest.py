import numpy as np
from collections import Counter, defaultdict
import pandas as pd
from numpy import genfromtxt
import array as arr
from sklearn.ensemble import RandomForestClassifier

used_features =[
        "Pclass",
        "Sex_cleaned",
        "Age",
        "SibSp",
        "Parch",
        "Embarked_cleaned"
    ]

def dataformatTest():
    data = pd.read_csv("test.csv")

    #categorical to numerical
    data["Sex_cleaned"]=np.where(data["Sex"]=="male",0,1)
    data["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0,
                                      np.where(data["Embarked"]=="C",1,
                                               np.where(data["Embarked"]=="Q",2,3)
                                              )
                                     )

    #converting age into categorical data
    data["Age"] = np.where(data["Age"]<25,0,np.where(data["Age"]>50,2,1))

    # remove NAN values

    data=data[[
        "Pclass",
        "Sex_cleaned",
        "Age",
        "SibSp",
        "Parch",
        "Embarked_cleaned"
    ]].dropna(axis=0, how='any')
    return data


def dataformatTrain():
    data = pd.read_csv("train.csv")

    #categorical to numerical
    data["Sex_cleaned"]=np.where(data["Sex"]=="male",0,1)
    data["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0,
                                      np.where(data["Embarked"]=="C",1,
                                               np.where(data["Embarked"]=="Q",2,3)
                                              )
                                     )

    #converting age into categorical data
    data["Age"] = np.where(data["Age"]<25,0,np.where(data["Age"]>50,2,1))

    # remove NAN values

    data=data[[
        "Survived",
        "Pclass",
        "Sex_cleaned",
        "Age",
        "SibSp",
        "Parch",
        "Embarked_cleaned"
    ]].dropna(axis=0, how='any')
    return data


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



if __name__ == "__main__":
    X_train = dataformatTrain()
    X_test = dataformatTest()
    trained_model = random_forest_classifier(X_train[used_features], X_train["Survived"])
    predictions = trained_model.predict(X_test[used_features])
    print(predictions)