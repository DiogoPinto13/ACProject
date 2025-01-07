import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Input
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.metrics import Recall, Precision
#from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import mae
import os
import pickle
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import threading
import os
from tensorflow.keras.callbacks import EarlyStopping

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

def preProcessSecondData(trainPath, dfTarget):
    """Function to read the images and return a dataframe"""
    df = pd.read_csv(trainPath, header=None)
    #split the df according to normal and abnormal
    anomalyImgIndex = dfTarget.index[dfTarget['TARGET'] == 1].tolist()
    normalIndex = dfTarget.index[dfTarget['TARGET'] == 0].tolist()

    #print(anomalyImgIndex)
    #print(normalIndex)  
    anomalyImg = df.loc[(anomalyImgIndex)]
    normalImg = df.loc[normalIndex]

    return pd.DataFrame(normalImg), pd.DataFrame(anomalyImg), df

def startTrain(optionFunction, optionString, dfTrain, dfTargetTrain, dfTest, dfTargetTest):
    accuracyList = list()
    precisionList = list()
    recallList = list()
    f1ScoreList = list()
    iteration = 0

    for i in range(5):
        iteration += 1
        optionFunction(dfTrain, dfTargetTrain, dfTest, dfTargetTest, accuracyList, precisionList, recallList, f1ScoreList)
        writeResults(iteration, optionString, accuracyList, precisionList, recallList, f1ScoreList)
    
    print("Accurancy: ", accuracyList)
    print("Precision: ", precisionList)
    print("Recall: ", recallList)
    print("F1Score: ", f1ScoreList)

def trainNN(dfTrain, dfTargetTrain, dfTest, dfTargetTest, accuracyList, precisionList, recallList, f1ScoreList):
    print("Using neural network")
    print(dfTrain.shape[1])
    model = Sequential()

    model.add(Dense(dfTrain.shape[1], activation="relu"))
    model.add(Dense(dfTrain.shape[1], activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(3, activation="relu"))
    model.add(Dense(1, activation="linear"))

    earlyStopping = EarlyStopping(patience=10, min_delta=1e-3, monitor="val_loss", restore_best_weights=True)
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    model.fit(dfTrain, dfTargetTrain, batch_size=16, epochs=100, verbose=1, callbacks=[earlyStopping])

    predictions = (model.predict(dfTest) > 0.5).astype(int)
    accuracyList.append(accuracy_score(dfTargetTest, predictions))
    
    cm    = confusion_matrix(dfTargetTest, predictions)
    TN, FP, FN, TP = cm.ravel()
    Precision    = TP/(TP+FP)
    Recall    = TP/(TP+FN)
    F1    = (2*Precision*Recall)/(Precision+Recall)
    
    precisionList.append(Precision)
    recallList.append(Recall)
    f1ScoreList.append(F1)


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
    """
    plt.figure(figsize=(5,5))
    plot_tree(model)
    plt.show()
    """
    accuracyList.append(accuracy_score(dfTargetTest, predictions))
    
    cm    = confusion_matrix(dfTargetTest, predictions)
    TN, FP, FN, TP = cm.ravel()
    Precision    = TP/(TP+FP)
    Recall    = TP/(TP+FN)
    F1    = (2*Precision*Recall)/(Precision+Recall)
    
    precisionList.append(Precision)
    recallList.append(Recall)
    f1ScoreList.append(F1)

def evaluateAutoencoder(autoencoder, df, dfTarget, threshold):
    reconstructed = autoencoder.predict(df)
    #loss = np.mean(np.abs(df - reconstructed), axis=1)
    loss = mae(reconstructed, df)
    predictions = (loss > threshold)
    
    #print("Predictions:", predictions)
    
    cm = confusion_matrix(dfTarget, predictions)
    trueNegatives, falsePositives, falseNegatives, truePositives = cm.ravel()
    total = trueNegatives + falsePositives + falseNegatives + truePositives
    
    accuracy = (truePositives + trueNegatives) / total
    precision = (truePositives) / (truePositives + falsePositives)
    recall = (truePositives) / (truePositives + falseNegatives)
    f1Score = (2 * precision * recall) / (precision + recall)
    print("accuracy = " + str(accuracy))
    print("precision = " + str(precision))
    print("recall = " + str(recall))
    print("f1Score = " + str(f1Score))
    
    return predictions

def trainImg(dfImgNormal, dfImgAnomaly, dfImg, dfTarget):
    accuracyList = list()
    precisionList = list()
    recallList = list()
    f1ScoreList = list()

    dfImgNormalTrain, dfImgNormalTest = train_test_split(dfImgNormal, test_size=0.30, random_state=45, shuffle=True)

    inputDim = dfImgNormal.shape[-1]
    latentDim = 32

    autoencoder = tf.keras.Sequential([
        #ENCODER
        layers.Input(shape=(inputDim,)),
        layers.Reshape((inputDim, 1)),  # Reshape to 3D for Conv1D
        layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2, padding="same"),
        layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2, padding="same"),
        layers.Conv1D(latentDim, 3, strides=1, activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2, padding="same"),

        #DECODER
        layers.Conv1DTranspose(latentDim, 3, strides=1, activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(inputDim)
    ])

    earlyStopping = EarlyStopping(patience=5, min_delta=1e-3, monitor="loss", restore_best_weights=True)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mae")
    autoencoder.fit(dfImgNormalTrain, dfImgNormalTrain, batch_size=16, epochs=100, verbose=1, callbacks=[earlyStopping])

    #evaluation
    trainMae = autoencoder.evaluate(dfImgNormalTrain, dfImgNormalTrain, verbose=0)
    testMae = autoencoder.evaluate(dfImgNormalTest, dfImgNormalTest, verbose=0)
    anomalyMae = autoencoder.evaluate(dfImgAnomaly, dfImgAnomaly, verbose=0)

    print("Training dataset error: ", trainMae)
    print("Testing dataset error: ", testMae)
    print("Anomaly dataset error: ", anomalyMae)

    dfImgNormalTrainTarget = np.zeros(dfImgNormalTrain.shape[0])
    dfImgNormalTestTarget = np.zeros(dfImgNormalTest.shape[0])
    dfImgAnomalyTarget = np.ones(dfImgAnomaly.shape[0])

    #compute the threshold
    reconstructed = autoencoder.predict(dfImgNormal, verbose=False)
    loss = mae(reconstructed, dfImgNormal)
    threshold = np.mean(loss) + 0.5*np.std(loss)

    #evaluate the model's performance
    print("Evaluating normal data used for training")
    evaluateAutoencoder(autoencoder, dfImgNormalTrain, dfImgNormalTrainTarget, threshold)
    
    print("Evaluating normal data used for testing")
    evaluateAutoencoder(autoencoder, dfImgNormalTest, dfImgNormalTestTarget, threshold)
    
    print("Evaluating abnormal data")
    evaluateAutoencoder(autoencoder, dfImgAnomaly, dfImgAnomalyTarget, threshold)
    
    predictions = evaluateAutoencoder(autoencoder, dfImg, dfTarget, threshold)
    
    return predictions

def writeResults(iteration, option, accuracyList, precisionList, recallList, f1ScoreList):
    filePath = str(option) + ".txt"
    
    #append mode
    with open(filePath, 'a') as file:
        file.write("Iteration: {}\n".format(iteration))
        file.write("Accuracy Mean: {}\n".format(np.mean(accuracyList)))
        file.write("Accuracy Std: {}\n".format(np.std(accuracyList)))
        
        file.write("Precision Mean: {}\n".format(np.mean(precisionList)))
        file.write("Precision Std: {}\n".format(np.std(precisionList)))
        
        file.write("Recall Mean: {}\n".format(np.mean(recallList)))
        file.write("Recall Std: {}\n".format(np.std(recallList)))

        file.write("F1Score Mean: {}\n".format(np.mean(f1ScoreList)))
        file.write("F1Score Std: {}\n".format(np.std(f1ScoreList)))


def main():
    options = [["Neural Network", trainNN], ["Decision Tree", trainDT]]
    dfData, dfTarget = preProcessFirstData("COVID_numerics.csv")
    dfImgNormal, dfImgAnomaly, dfImg = preProcessSecondData("COVID_IMG.csv", dfTarget)
    print(dfData.head(5))
    print(dfTarget.head(5))

    predictions = trainImg(dfImgNormal, dfImgAnomaly, dfImg, dfTarget)
    predictions = tf.cast(predictions,tf.float32).numpy()
    
    print("Predictions:", predictions)
    
    #add the predictions of the ECG
    dfData[dfData.columns.size] = predictions
    dfTrain, dfTest, dfTargetTrain, dfTargetTest = train_test_split(dfData, dfTarget, test_size=0.2, random_state=42)

    print("dfData:",dfData.head(5))
    print("dfTrain: ", dfTrain.head(5))
    hOptions = list()
    
    for i in range(len(options)):
        hOptions.append(threading.Thread(target=startTrain, args=(options[i][1], options[i][0], dfTrain, dfTargetTrain, dfTest, dfTargetTest)))
    
    for i in range(len(hOptions)):
        hOptions[i].start()
    for i in range(len(hOptions)):
        hOptions[i].join()
        

if __name__ == "__main__":
    main()