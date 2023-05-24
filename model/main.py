
import pandas as pd
import numpy as np
import os
import pickle5 as pickle



#
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import f1_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
# from sklearn.metrics import log_loss
# from sklearn.metrics import brier_score_loss
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import explained_variance_score
# from sklearn.metrics import r2_score
# from sklearn.metrics import median_absolute_error

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import streamlit as st

data_path = '../data/data.csv'


def create_model(bc_data):
    X = bc_data.drop(columns=['diagnosis'], axis=1)
    Y = bc_data['diagnosis']

    # scaled data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # model

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    return model, scaler



def get_clean_data(bc_data):
    """
    Reads in the data from the specified path and returns a pandas dataframe.
    """
    df = pd.read_csv(bc_data)

    df = df.drop(columns=['Unnamed: 32', 'id'], axis=1)

    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})



    return df


def test_model(model):
    """
    Tests the model on the test data.
    """
    data = get_clean_data(data_path)
    X = data.drop(columns=['diagnosis'], axis=1)
    Y = data['diagnosis']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Y_pred = model.predict(X)

    print(classification_report(Y, Y_pred))
    print('Accuracy of our model: ',  accuracy_score(Y, Y_pred))




# def add_sidebar():
#     create_sliders(df)




def main():

    data = get_clean_data(data_path)
    print(data.info())

    model, scaler = create_model(data)


    # test_model(model)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)


    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)



if __name__ == '__main__':
    main()

