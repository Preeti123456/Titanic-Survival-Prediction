import numpy as np
from collections import Counter, defaultdict
import pandas as pd
from numpy import genfromtxt
import array as arr

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

 
def occurrences(list1):
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob
 
def naive_bayes(training, outcome, new_sample):
    classes     = np.unique(outcome)
    rows, cols  = np.shape(training)
    likelihoods = {}
    for cls in classes:
        likelihoods[cls] = defaultdict(list)
    #Probability of each class  
    class_probabilities = occurrences(outcome)
  
    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset      = training[row_indices, :]
        r, c        = np.shape(subset)
      
        for j in range(0,c):
            likelihoods[cls][j] += list(subset[:,j])
      
    for cls in classes:
        for j in range(0,cols):
             likelihoods[cls][j] = occurrences(likelihoods[cls][j])  
  
    results = {}
    for cls in classes:
         class_probability = class_probabilities[cls]
         for i in range(0,len(new_sample)):
             relative_values = likelihoods[cls][i]
             if new_sample[i] in relative_values.keys():
                 class_probability *= relative_values[new_sample[i]]
             else:
                 class_probability *= 0
             results[cls] = class_probability
    if(results[0]>results[1]):
        return 0
    else:
        return 1
  
if __name__ == "__main__":
    dTrain = dataformatTrain()
    training   = np.asarray(dTrain[used_features])
    outcome    = np.asarray(dTrain["Survived"])
    dTest = dataformatTrain()
    dTest = np.asarray(dTest[used_features])
    ans = [5]
    for new_sample1 in dTest:
        a = naive_bayes(training, outcome, new_sample1)
        ans.append(a)
    ans.remove(5)
    print(ans)