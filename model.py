import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


# pre-processing
def pre_processing():
    df = pd.read_csv('Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv')

    lag_day = [1, 5, 15, 25]

    df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
    df.sort_values(by = 'Date', ignore_index=True, inplace=True)
    df = df.set_index('Date')
    df['Stock_trading_diff'] = df['Stock Trading'].pct_change()

    for lag in lag_day:
        col_name = f"price_change_{str(lag)}"
        df[col_name] = df['Close'].pct_change(lag)

    for lag in lag_day:
        col_name = f"SMA_{str(lag)}"
        df[col_name] = df['Close'].rolling(lag).mean()

    for lag in lag_day:
        col_name = f"EWMA_{str(lag)}"
        df[col_name] = df['Close'].ewm(span = lag, adjust = True).mean()

    df['high_low_pred'] = np.where(df['Close'] > df['Close'].shift(-1), 0 , 1)

    # target and feature cols
    target_col_logistic = 'high_low_pred'
    feature_col = [col for col in df.columns if col not in target_col_logistic]



    df.drop(df.index[:25], inplace=True)
    df.drop(df.tail(1).index, inplace = True)

    return df, feature_col, target_col_logistic

def standadization():
    df, feature_col, target_col_logistic = pre_processing() 

    #Standard scaling
    scaler = StandardScaler()
    scaler.fit(df[feature_col])
    df[feature_col] = scaler.transform(df[feature_col])
    return df, feature_col, target_col_logistic, scaler

def model():
    # 50% of the data for training
    df, feature_col, target_col_logistic, _ = standadization()
    training_df = df.iloc[:int(.5*df.shape[0])]

    # 30% of the data for validation
    validation_df = df.iloc[int(.5*df.shape[0]) : int(.8*df.shape[0])]

    # 80% of the data for training if the training/validation stratigy turns out to be less optimal
    ex_training_df = df.iloc[:int(.8*df.shape[0])]

    # 20% of the data for testing. Doing this before standardscaling so I can use it in request.py
    testing_df = df.iloc[int(.8*df.shape[0]) : ]


    # Model

    log_model = LogisticRegression()

    log_model.fit(training_df[feature_col], training_df[target_col_logistic])

    search = GridSearchCV(log_model, { 'solver': ['newton-cg', 'lbfgs','liblinear'], 
                                'tol': [0.001,.009,0.01],
                                'C':np.logspace(-4, 4, 20), 
                                'max_iter':[50,100,150,200],
    })

    search.fit(validation_df[feature_col], validation_df[target_col_logistic])

    pickle.dump(search, open('model.pkl','wb'))


    predictions = search.predict(testing_df[feature_col])

    print('Accuracy: ', round(accuracy_score(predictions, testing_df[target_col_logistic]), 3))
    print('Precision: ', round(precision_score(predictions, testing_df[target_col_logistic]), 3))
    print('Recall: ', round(recall_score(predictions, testing_df[target_col_logistic]), 3))
    print('F1: ', round(f1_score(predictions, testing_df[target_col_logistic]), 3))