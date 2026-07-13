import pandas as pd
import numpy as np

def load_and_prepare(train_path="data/mitbih_train.csv", test_path="data/mitbih_test.csv"):
    train = pd.read_csv(train_path, header=None)
    test  = pd.read_csv(test_path, header=None)
    
    # Reshape features for 1D CNN: (samples, time_steps, channels)
    X_train = train.iloc[:, :187].values.reshape(-1, 187, 1)
    y_train = train.iloc[:, 187].values.astype(int)
    
    X_test  = test.iloc[:, :187].values.reshape(-1, 187, 1)
    y_test  = test.iloc[:, 187].values.astype(int)
    
    return X_train, y_train, X_test, y_test
