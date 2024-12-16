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

def changeDataset(df):
    """This function is meant to remove dropna values and remove spaces in strings"""
    #print(f'Nº null values: {df.isna().sum().sum()}')
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
    
    print(df.head(5))
    featuresToDelete = ["MARITAL STATUS"]
    df = deleteFeatures(df, featuresToDelete)

    dfTargetTrain = pd.DataFrame(df.pop('TARGET'))
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


def trainNN(dfData, dfTarget):
    print("Using neural network")

def trainDT(dfData, dfTarget):
    print("Using Decision Trees")

def writeResults(option, accuracyList, precisionList, recallList, f1ScoreList):
    pass

def main():
    options = [["Neural Network", trainNN], ["Decision Tree", trainDT]]
    
    dfData, dfTarget = preProcessFirstData("COVID_numerics.csv")
    for i in range(len(options)):
        accuracyList = list()
        precisionList = list()
        recallList = list()
        f1ScoreList = list()

        for j in range(30):
            options[i][1](dfData, dfTarget)
            writeResults(options[i][0], accuracyList, precisionList, recallList, f1ScoreList)


if __name__ == "__main__":
    main()