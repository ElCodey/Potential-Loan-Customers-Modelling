from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


def logistic_model(x_train, x_test, y_train, y_test):
    model = LogisticRegression()
    fit_model = model.fit(x_train, y_train)
    y_pred = fit_model.predict(x_test)
    model_accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    prec_score = precision_score(y_test, y_pred)
    return model_accuracy, conf_matrix, prec_score

def cross_validation_log(df, x):
    model = LogisticRegression()
    y = df["personal_loan"]
    x = df[x]
    scores = cross_val_score(model, x, y, cv=10)
    return scores.mean()

def naive_bayes(x_train, x_test, y_train, y_test):
    model = GaussianNB()
    fit_model = model.fit(x_train, y_train)
    y_pred = fit_model.predict(x_test)
    model_accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    prec_score = precision_score(y_test, y_pred)
    return model_accuracy, conf_matrix, prec_score

def cross_validation_bayes(df, x):
    model = GaussianNB()
    y = df["personal_loan"]
    x = df[x]
    scores = cross_val_score(model, x, y, cv=10)
    return scores.mean()