'''
Copyright The Jackson Laboratory, 2022, 2023
authors: Jim Peterson, Abed Ghanbari

This script implements the training of classificaiton models for MMAR

To Do:
    - de-lint
'''
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ClassifierNames(Enum):
    '''
    The classifier names class
    Orders of the classifiers in the list are important
    Any new classifier should be added to the end of the list
    It should also be added to the MotionFrameDetector and Parameters class in same order
    '''
    LogisticRegression = 'Logistic Regression'
    SVM = 'SVM'
    RandomForest = 'Random Forest'
    AdaBoostedTree = 'AdaBoosted decision tree'
    GradientBoostedTree = 'Gradient Boosted Decision Tree'
    GaussianNaiveBayes = 'Gaussian Naive Bayes'

    @classmethod
    def get_index(cls, type):
        return list(cls).index(type)

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class MotionFrameDetector:
    '''
    Trains models to find bad frames using following methods:
        'Logistic Regression',
        'SVM',
        'Random Forest',
        'AdaBoosted decision tree',
        'Gradient Boosted Decision Tree',
        'Gaussian Naive Bayes',

    Parameters
    ----------
    X: numpy array of shape (num of observations, num of features)
        The feature matrix
    Y: numpy array of shape (num of observations,)
        The label vector
    train_balanced: bool, default: False
        Whether to balance the training data
    selected_methods: list
        A list of booleans to select which classifiers to train and test
    params: list of dictionaries
        A list of dictionaries to specify the parameters for each classifier

    Attributes
    ----------
    X_train, X_test : numpy array
        The feature matrix of train and test
    Y_train, Y_test : numpy array
        The label vector of train and test
    upsample_factor: float
        if train_balanced, upsamples the class with lower number of observation to
        `upsample_factor` * number of observations in larger class
    downsample_factor: float
        if train_balanced, downsamples the class with larger number of observation to
        `downsample_factor` * number of observations in smaller class
    nfold : int
        Number of folds in GridSearchCV
    classifier_accuracies : list
    classifier_predictions : list
    classifier_probabilities : list
    classifier_confusion_matrices : list
    classifier_classification_reports : list
    classifier_roc_auc_scores : list
    classifier_roc_curves : list
    classifier_balanced_accuracy : list
    classifier_recall_score : list

    Methods
    -------
    train()
        Trains classifier using exhaustive search over parameters
    print_results()
        Prints the accuracy, recall score, balanced accuracy, and ROC AUC score for each classifier
    plot_confusion_matrices()
        Plots the confusion matrices for each classifier
    plot_roc_curves()
        Plots the ROC curves for each classifier
    plot_roc_auc_scores()
        Plots the ROC AUC scores for each classifier
    plot_balanced_scores()
        Plots the balanced accuracy scores for each classifier

    '''
    def __init__(self, X, Y,
                 train_balanced=True,
                 selected_methods=None,
                 params=None,
                 classifier_name=None,
                 search_param=True):

        self.X = X
        self.Y = Y

        self.train_balanced = train_balanced

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=23)

        self.upsample_factor = .4
        self.downsample_factor = .6

        self.nfold = 5

        self.classifiers = [
            LogisticRegression(),
            SVC(probability=True),
            RandomForestClassifier(),
            AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)),
            GradientBoostingClassifier(),
            GaussianNB()
            ]

        self.classifier_names = ClassifierNames

        if classifier_name is not None:
            if classifier_name=='all':
                selected_methods = [True]*len(self.classifier_names)
            else:
                if classifier_name in self.classifier_names.list():
                    selected_methods = [False]*len(self.classifier_names)
                    _index = [idx for idx, s in enumerate(self.classifier_names.list()) if classifier_name in s][0]
                    selected_methods[_index] = True
                else:
                    raise Exception(f'Classifier name: {classifier_name} not found in {self.classifier_names.list()}')

        if params:
            self.params=params
        else:
            if search_param:
                par = Parameters.params
            else:
                par = Parameters.best_params
            self.params = [
                # 1 - Logistic Regression
                    {
                        "penalty" : par['penalty'],
                        "tol" : par['tol'],
                        "max_iter" : par['max_iter_log'],
                    },
                # 2 - SVM
                    {
                        "C": par['C'],
                        "gamma": par['gamma'],
                        "kernel" : par['kernel'],
                        "degree" : par['degree'],
                        "max_iter" : par['max_iter_svm'],
                    },
                # 3 - Random Forest
                    {
                        'n_estimators': par['n_estimators_rf'],
                        'max_depth': par['max_depth_rf'],
                        'min_samples_leaf': par['min_samples_leaf'],
                        'class_weight': par['class_weight'],
                    },
                # 4 - AdaBoosted decision tree
                    {
                        "algorithm" : par['algorithm_ada'],
                        "n_estimators" : par['n_estimators_ada'],
                    },
                # 5 - Gradient Boosted Decision Tree
                    {
                        "learning_rate" : par['learning_rate'],
                        "n_estimators" : par['n_estimators_gb'],
                        "max_depth" : par['max_depth_gb'],
                    },
                # 6 - Gaussian Naive Bayes
                    {
                        "var_smoothing": par['var_smoothing'],
                    },
                ]

        if selected_methods:
            self.classifiers = [i for idx, i in enumerate(self.classifiers) if selected_methods[idx]]
            self.classifier_names = [i for idx, i in enumerate(self.classifier_names) if selected_methods[idx]]
            self.params = [i for idx, i in enumerate(self.params) if selected_methods[idx]]

        self.classifier_accuracies = []
        self.classifier_predictions = []
        self.classifier_probabilities = []
        self.classifier_confusion_matrices = []
        self.classifier_classification_reports = []
        self.classifier_roc_auc_scores = []
        self.classifier_roc_curves = []
        self.classifier_balanced_accuracy = []
        self.classifier_recall_score = []


    def train(self):
        '''
        Trains classifier using exhaustive search over parameters
        Best estimator will be selcted to calculate metrics

        Returns
        -------
        GridSearchCV class after fitting all models
        with associated:
            cv_results_,
            best_estimator_,
            best_score_,
            best_params_,
            scorer_,
            ...
        '''
        if self.train_balanced:
            X, y = self.balance_data(self.X_train, self.Y_train)
        else:
            X, y = self.X_train, self.Y_train

        cv_results = []
        for classifier, params in zip(self.classifiers, self.params):
            print(f"Training {classifier.__class__.__name__}")
            print('-'*50)
            print(f"Parameters: {params}")

            pipeline_estimator = skPipeline([('scaler',  StandardScaler()), ('Classifier', classifier)])
            grid_params = {'Classifier__'+k: v for k, v in params.items()}

            gridSCV = GridSearchCV(pipeline_estimator, grid_params, scoring='recall_macro', refit='recall_macro', cv=self.nfold, n_jobs=-1)
            gridSCV.fit(X, y)
            classifier = gridSCV.best_estimator_

            cv_results.append(gridSCV)

            self.classifier_accuracies.append(classifier.score(self.X_test, self.Y_test))
            self.classifier_predictions.append(classifier.predict(self.X_test))
            self.classifier_probabilities.append(classifier.predict_proba(self.X_test))
            self.classifier_confusion_matrices.append(confusion_matrix(self.Y_test, self.classifier_predictions[-1]))
            self.classifier_classification_reports.append(classification_report(self.Y_test, self.classifier_predictions[-1]))
            self.classifier_roc_auc_scores.append(roc_auc_score(self.Y_test, self.classifier_probabilities[-1][:, 1]))
            self.classifier_roc_curves.append(roc_curve(self.Y_test, self.classifier_probabilities[-1][:, 1]))
            self.classifier_balanced_accuracy.append(balanced_accuracy_score(self.Y_test, self.classifier_predictions[-1]))
            self.classifier_recall_score.append(recall_score(self.Y_test, self.classifier_predictions[-1]))

        return cv_results

    def balance_data(self, X, y):
        '''
        balance the minority class
        ref: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
        '''
        # over = SMOTE(sampling_strategy=0.5)
        over = ADASYN(sampling_strategy=self.upsample_factor)
        under = RandomUnderSampler(sampling_strategy=self.downsample_factor)
        steps = [('o', over), ('u', under)]
        pipeline = imbPipeline(steps=steps)
        X, y =pipeline.fit_resample(X, y)
        return X, y

    def print_results(self):
        for classifier_name, classifier_accuracy, recall_score, classifier_balanced_accuracy, classifier_roc_auc_score in zip(self.classifier_names, self.classifier_accuracies, self.classifier_recall_score, self.classifier_balanced_accuracy, self.classifier_roc_auc_scores):
            print(f'Accuracy of {classifier_name}: {(classifier_accuracy*100):.2f}%')
            print(f'Recall score of {classifier_name}: {(recall_score*100):.2f}%')
            print(f'Balanced Accuracy of {classifier_name}: {(classifier_balanced_accuracy*100):.2f}%')
            print(f'ROC AUC score of {classifier_name}: {(classifier_roc_auc_score*100):.2f}%')
            print(f'Confusion matrix of {classifier_name}:')
            print(pd.DataFrame(self.classifier_confusion_matrices[-1], index=['Actual Accepted Frames', 'Actual Rejected Frames'], columns=['Predicted Accepted Frames', 'Predicted Rejected Frames']))

            print('\n')

    def plot_confusion_matrices(self):
        plt.figure(dpi=75, figsize=(12,12))

        k = 1
        for classifier_name, classifier_confusion_matrix in zip(self.classifier_names, self.classifier_confusion_matrices):
            plt.subplot(4,2,k)
            plt.imshow(classifier_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion matrix for {classifier_name}')

            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['0', '1'], rotation=45)
            plt.yticks(tick_marks, ['0', '1'])

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            k+=1
        plt.show()

    def plot_roc_curves(self):
        for classifier_name, classifier_roc_curve in zip(self.classifier_names, self.classifier_roc_curves):
            plt.plot(classifier_roc_curve[0], classifier_roc_curve[1], label=classifier_name.value)
        plt.plot([0, 1], [0, 1], 'k--', label='')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curves')
        plt.legend()
        plt.savefig("roc.png")
        plt.draw()
        #plt.close()


    def plot_roc_auc_scores(self):
        plt.figure()
        plt.bar(range(len(self.classifier_roc_auc_scores)), self.classifier_roc_auc_scores, align='center')
        plt.xticks(range(len(self.classifier_roc_auc_scores)), self.classifier_names, rotation=45, ha='right')
        plt.title('ROC AUC score')
        plt.ylim((0.75, 1))
        plt.show()

    def plot_balanced_scores(self):
        plt.figure()
        plt.bar(range(len(self.classifier_balanced_accuracy)), self.classifier_balanced_accuracy, align='center')
        plt.xticks(range(len(self.classifier_balanced_accuracy)), self.classifier_names, rotation=45, ha='right')
        plt.title('Balanced Accuracy')
        plt.ylim((0.5, 1))
        plt.show()

class Parameters:
    params = {
            'penalty': ['l2'],
            'max_iter_log': [1000, 8000],
            'tol': [1e-4, 1e-3, 1e-1],

            'C': [0.1, 1, 10, 50, 100],
            'gamma': ['auto'],
            'kernel': ['rbf'],
            'degree': [3, 5, 7, 10],
            'max_iter_svm': [6000],

            'n_estimators_rf': [10, 100],
            'max_depth_rf': [None, 2, 10],
            'min_samples_leaf': [.1, .2, 1],
            'class_weight': ['balanced', None],

            'algorithm_ada': ['SAMME', 'SAMME.R'],
            'n_estimators_ada': [10, 50, 100],

            'learning_rate': [0.01, 0.1, 0.5, 1],
            'n_estimators_gb': [10, 50, 100],
            'max_depth_gb': [None, 2, 5, 10, 50],

            'var_smoothing' : np.logspace(-10, -1, 5, base=2),
        }
    best_params = {
            'penalty': ['l2'],
            'max_iter_log': [5000],
            'tol': [1e-4],

            'C': [10],
            'gamma': ['auto'],
            'kernel': ['rbf'],
            'degree': [3],
            'max_iter_svm': [5000],

            'n_estimators_rf': [100],
            'max_depth_rf': [None],
            'min_samples_leaf': [1],
            'class_weight': ['balanced'],

            'algorithm_ada': ['SAMME.R'],
            'n_estimators_ada': [100],

            'learning_rate': [0.5],
            'n_estimators_gb': [100],
            'max_depth_gb': [10],

            'var_smoothing' : [0.22],
    }
