import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from nltk.tokenize import word_tokenize
import numpy as np


def build_dataset(path, num_class_samples=-1, rnd_state=42):
    df = pd.read_excel(path, engine='openpyxl')
    df = df[df['3_label_majority_answer'] == 'Agree']
    df = df.dropna(subset=['tweet'])
    if num_class_samples != -1:
        df = df.sample(n=min(len(df), num_class_samples), replace=False, random_state=rnd_state)
    return df.T.to_dict()

def tune_logistic_regression(X_train, Y_train):
    model = LogisticRegression()
    param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
                  'penalty': ['l2'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'max_iter': [5000]}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return best_model

def tune_svm(X_train, Y_train):
    model = SVC()
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  
        'degree': [2, 3, 4, 5, 6, 7, 8, 9],
        'gamma': ['scale', 'auto'], 
        }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return best_model

def evaluate(Y_test, y_pred):
    print('Precision: ', precision_score(Y_test, y_pred))
    print('Recall: ', recall_score(Y_test, y_pred))
    print('F1_score: ', f1_score(Y_test, y_pred))
    print('accuracy: ', accuracy_score(Y_test, y_pred))

def preprocess_text(text, language='french'):
    return word_tokenize(text.lower(), language=language)

def text_to_word2vec(text, model):
    words = preprocess_text(text)
    vectors = [model[word] for word in words if word in model]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)