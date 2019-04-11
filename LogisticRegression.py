# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:44:39 2019

@author: Ahmet Hasim
"""


class LogisticRegression:
    def __init__(self, alpha=0.01, num_iter=100000, fit_intercept=True):
        self.alpha = alpha
        self.num_iter = num_iter # to ensure that we stop iteration
        self.fit_intercept = fit_intercept

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))  # for theta0, we need a column with all values=1
        return np.concatenate((intercept, X), axis=1) # and we concetenate this two matrices

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __log_likelihood(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X_Train, y_Train, X_Test, y_Test, batch_size):
        if self.fit_intercept:
            X_Train = self.__add_intercept(X_Train) # for theta0, we need extra column

        # results will be our predictions
        results = []
        # weights initialization
        self.theta = np.zeros(X_Train.shape[1])

        newLoss = 0.0
        for j in range(self.num_iter):
            loss = 0.0
            m=len(y_Train)
            for i in range(0, m, batch_size):
                X_piv = X_Train[i:i + batch_size] # mini-batch 
                y_piv = y_Train[i:i + batch_size] #mini batch 
                z = np.dot(X_piv, self.theta) # theta0 + theta1*x1 + theta2*x2 +.... + thetai *xi
                h = self.__sigmoid(z) # h function
                gradient = np.dot(X_piv.T, (h - y_piv)) / y_piv.size # gradient
                self.theta -= self.alpha * gradient
                loss += self.__log_likelihood(h, y_piv)

            if (np.absolute(loss - newLoss) < 0.001):
                preds = self.predict(X_Test, 0.5)
                p = self.prediction_accuracy(y_Test, preds)
                print(" !! LAST CALL !! Prediction accuracy is {}".format(p))
                results.append(p)
                return results

            newLoss = loss
            preds = self.predict(X_Test, 0.5)
            p = self.prediction_accuracy(y_Test, preds)
            print("Prediction accuracy is {}".format(p))
            results.append(p)


        return results

    def predict_prob(self, X, threshold=0.5):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold

    def prediction_accuracy(self, predicted_labels, original_labels):
        count = 0
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == original_labels[i]:
                count += 1
        # print count, len(predicted_labels)
        return float(count) / len(predicted_labels)


import numpy as np
import pandas as pd  # just used for creating dataframe (permission was taken)
import matplotlib.pyplot as plt


#-----------------MODEL FOR IONOSPHERE-------------------
filename = 'ionosphere.data'
df = pd.read_csv(filename, sep="\n", quotechar=None, quoting=3, header=None, delimiter=',')

# We changes char values to numeric values
new_map = {"b": 0, "g": 1}  # 0 for blue , 1 for green
df[34] = df[34].map(new_map)
df2 = df.sample(frac=1).reset_index(drop=True) # SHUFFLE ALL DATA ROWS

#seperating the datas as train and test

rate = int(((len(df2))*80)/100)
train = df2[:rate]
test = df2[rate:]

dataset_Train = train.values
dataset_Test = test.values

X_Train = dataset_Train[:, :-1] # adding features to X
y_Train = dataset_Train[:, -1] # adding features to y

X_Test = dataset_Test[:, :-1] # adding features to X
y_Test = dataset_Test[:, -1] # adding features to y


model = LogisticRegression(alpha=0.1, num_iter=1000)

pred1 = model.fit(X_Train, y_Train, X_Test, y_Test, 32)

plt.figure(figsize=(9, 6))
plt.plot(pred1)
plt.xlabel("Iterations")
plt.ylabel("Prediction Accuracy")
plt.title("IONOSPHERE DATAS")
plt.show()

# ------------------------------------------------------------------------------------------------------------
#-----------------MODEL FOR SONAR-------------------

filename = 'sonar.all-data'
df = pd.read_csv(filename, sep="\n", quotechar=None, quoting=3, header=None, delimiter=',')

# We changes char values to numeric values
new_map = {"R": 0, "M": 1}  # 0 for ROCKS , 1 for MINES
df[60] = df[60].map(new_map)
df2 = df.sample(frac=1).reset_index(drop=True)  # SHUFFLE ALL DATA ROWS


#seperating the datas as train and test
rate = int(((len(df2))*80)/100)
train = df2[:rate]
test = df2[rate:]

dataset_Train = train.values
dataset_Test = test.values

X_Train = dataset_Train[:, :-1] # adding features to X
y_Train = dataset_Train[:, -1] # adding outcome to y

X_Test = dataset_Test[:, :-1] # adding features to X
y_Test = dataset_Test[:, -1] # adding outcome to y

model2 = LogisticRegression(alpha=0.1, num_iter=1000)

pred2 = model2.fit(X_Train, y_Train, X_Test, y_Test, 32)

plt.figure(figsize=(9, 6))
plt.plot(pred2)
plt.xlabel("Iterations")
plt.ylabel("Prediction Accuracy")
plt.title("SONAR DATAS")
plt.show()