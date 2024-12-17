import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Recall, Precision
#from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score
from tensorflow.keras.losses import CategoricalCrossentropy
import os
import pickle
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
import threading
import os

def changeDataset(df):
    """This function is meant to remove dropna values and remove spaces in strings"""
    #print(f'Nº null values: {df.isna().sum().sum()}')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.rename(columns=lambda x: x.strip() if isinstance(x, str) else x)    # REMOVE SPACES IN STRING
    df = df.dropna().reset_index(drop=True)                                     # REMOVE NULL VALUE
    return df

def deleteFeatures(df: pd.DataFrame, featuresList):
    """function to delete a list of features from a DataFrame"""
    feautres = [feature for feature in featuresList if feature in df.columns]
    df.drop(columns=feautres, inplace=True)
    return df

def preProcessFirstData(trainPath):
    """Function to clean the CSV and prepare for training"""

    df = pd.read_csv(trainPath)
    df = changeDataset(df)
    featuresToDelete = ["MARITAL STATUS"]
    df = deleteFeatures(df, featuresToDelete)
    
    dfTargetTrain = pd.DataFrame(df.pop('TARGET')).astype(int)
    #eliminate possible other missing values and replace them with NaN
    
    df = df.apply(pd.to_numeric, errors='coerce')
    #remove non numeric features
    df = df.select_dtypes(include=[np.number])
    
    maxValue = np.finfo(np.float32).max
    minValue = np.finfo(np.float32).min
    df.replace([np.inf, -np.inf], [maxValue, minValue], inplace=True)

    imputer = SimpleImputer(strategy='most_frequent')
    dfImputed = imputer.fit_transform(df)

    scaler = MinMaxScaler()
    dfNormalized = scaler.fit_transform(dfImputed)

    return pd.DataFrame(dfNormalized), pd.DataFrame(dfTargetTrain)


def startTrain(optionFunction, optionString, dfTrain, dfTargetTrain, dfTest, dfTargetTest):
    accuracyList = list()
    precisionList = list()
    recallList = list()
    f1ScoreList = list()
    
    for i in range(5):
        optionFunction(dfTrain, dfTargetTrain, dfTest, dfTargetTest, accuracyList, precisionList, recallList, f1ScoreList)
        writeResults(optionString, accuracyList, precisionList, recallList, f1ScoreList)
    
    print(accuracyList)

def trainNN(dfTrain, dfTargetTrain, dfTest, dfTargetTest, accuracyList, precisionList, recallList, f1ScoreList):
    print("Using neural network")
    print(dfTrain.shape[1])
    model = Sequential()

    model.add(Dense(dfTrain.shape[1], activation="relu"))
    model.add(Dense(dfTrain.shape[1], activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    model.fit(dfTrain, dfTargetTrain, batch_size=32, epochs=10, verbose=1)

    predictions = (model.predict(dfTest) > 0.5).astype(int)  # Predict and threshold probabilities
    accuracyList.append(accuracy_score(dfTargetTest, predictions))


def trainDT(dfTrain, dfTargetTrain, dfTest, dfTargetTest, accuracyList, precisionList, recallList, f1ScoreList):
    print("Using Decision Trees")
    model = DecisionTreeClassifier(criterion='entropy', 
                                   splitter='best', 
                                   max_depth=3, 
                                   min_samples_split=4,
                                   min_samples_leaf=2,
                                   max_features=None, 
                                   random_state=42, 
                                   max_leaf_nodes=4)
    
    model.fit(dfTrain, dfTargetTrain)

    predictions = model.predict(dfTest)
    accuracyList.append(accuracy_score(dfTargetTest, predictions))


def writeResults(option, accuracyList, precisionList, recallList, f1ScoreList):
    pass

def main():
    options = [["Neural Network", trainNN], ["Decision Tree", trainDT]]
    
    dfData, dfTarget = preProcessFirstData("COVID_numerics.csv")
    print(dfData.head(5))
    print(dfTarget.head(5))

    dfTrain, dfTest, dfTargetTrain, dfTargetTest = train_test_split(dfData, dfTarget, test_size=0.2, random_state=42)

    hOptions = list()

    for i in range(len(options)):
        hOptions.append(threading.Thread(target=startTrain, args=(options[i][1], options[i][0], dfTrain, dfTargetTrain, dfTest, dfTargetTest)))

    for i in range(len(hOptions)):
        hOptions[i].start()
        hOptions[i].join()
        

if __name__ == "__main__":
    main()