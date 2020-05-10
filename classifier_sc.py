import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


basic_params_dict = {
    
    'X':['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    'y':['species_id'],
    'seed': 123,
    'test_size': 0.3
}

classifier_config_dict = {

    # Classifiers
   

    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'LogisticRegression':LogisticRegression(),
    'SVM':SVC()

 

}
gridsearch_model_params_dict = {
     # Hyperparameters
    'DecisionTreeClassifier': {'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)},
    'LogisticRegression': {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]},
    'SVM': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]}
    ]
}