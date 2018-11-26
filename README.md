# Titanic-Survival-Prediction
## INTRODUCTION
A machine learning project, implementing various machine learning algorithms.
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships. One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class. We try to analyse what sort of people were likely to survive.


## DATASET
File-train.csv 891 records
File- test.csv 418 records


# Naive Bayes algorithm (own implementation from scratch)
The dataset is taken, preprocessed to remove any missing values, converting the categorical values into numerical values and reduced the number of features. This is followed by splitting of the training dataset into 0.5 to chceck for accuracy of our model.
The probabilities of all unique classes is calculated and the classification of a dataset is done as per Bayes Theorem.
The accuracy comes out to be 76.45%.
File for Code- NaiveBayesL.py

## MACHINE LEARNING TECHNIQUES
# Naive Bayes algorithm (using library functions)
The dataset is pre-processed and cleaned for any missing values. Sklearn is used to import the GaussianNB function. 
The accuracy comes out to be 77.57%.
File for Code- NaivBayes.py

# Random Forest algorithm (using library functions)
The dataset is pre-processed and cleaned for any missing values. Sklearn is used to import the RandomForest function. 
The accuracy comes out to be 75.33%
File for Code- RandomForest.py

# Recurrent Neural network
The dataset is pre-processed and cleaned for any missing values. TensorFlow package is used to implement rnn.
The accuracy comes out to be 83.04%
File for Code- rnn.py


## CONCLUSION
Rnn Algorithm shows the best accuracy. Other algithms can be explored to iimprove the accuracy. It can further be improved by using large datasets to avoid any situation of Overfitting followed by a stratified k fold cross validation. 


## REFERENCES
https://www.kaggle.com/c/titanic
