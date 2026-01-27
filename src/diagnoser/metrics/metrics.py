from sklearn.metrics import accuracy_score, mean_squared_error

def classification_error(y_true, y_pred):
    
    return 1 - accuracy_score(y_true, y_pred)

def regression_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)