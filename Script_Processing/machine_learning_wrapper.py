from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.utils.extmath import density
from sklearn import metrics
import pandas as pd
import time
import eli5
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.base import clone
import math
import pickle
from Script_Processing.preprocessing_custom import full_preprocess
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

class MachineLearningWrapper:
    def __init__(self, target_names = None, refit = "average_precision"):
        self.universal_random_state = 42
        self.skf = StratifiedKFold(n_splits=5, random_state=self.universal_random_state, shuffle=True)
        self.n_cores = 1
        self.scoring_metrics = ['accuracy', 'balanced_accuracy',
                                'f1_weighted',
                                'precision_weighted', 'recall_weighted', 
                                'average_precision','roc_auc']
        self.refit = refit
        if self.refit not in self.scoring_metrics:
            self.scoring_metrics.append(self.refit)
        self.full_name = {}
        self.hyper_params = {}
        self.target_names = target_names
        self.get_clf_full_names()
        self.get_hyper_params()
        self.select_from_model = None

    def set_target_names(self, names):
        self.target_names = names

    def get_fresh_classifiers(self):
        '''
        Returns fresh instances of each classifier.
        '''
        return [(RidgeClassifier(), "ridgeclf"),
            (KNeighborsClassifier(n_jobs=self.n_cores), "knn"),
            (RandomForestClassifier(random_state = self.universal_random_state, n_jobs=self.n_cores), "rfc"),
            (LinearSVC(random_state = self.universal_random_state, max_iter=5000), "linsvm"),
            (SGDClassifier(loss='hinge', penalty='l2', n_jobs=self.n_cores, random_state = self.universal_random_state), "gdsvm"),
            (SGDClassifier(loss='log', n_jobs=self.n_cores, random_state = self.universal_random_state), "gdlr"),
            (LogisticRegression(random_state = self.universal_random_state, n_jobs=self.n_cores, max_iter=5000), "lr"),
            (MultinomialNB(), "mnb"),
            (ComplementNB(), "cnb"),
            (BernoulliNB(), "bnb"),
            (DecisionTreeClassifier(random_state = self.universal_random_state),"dectree"),
            (MLPClassifier(random_state = self.universal_random_state), "neuralnet")
           ]

    def get_clf_full_names(self):
        self.full_name["ridgeclf"] = "Ridge Classifier"
        self.full_name["pcpt"] = "Perceptron"
        self.full_name["knn"] = "K-Nearest Neighbors"
        self.full_name["rfc"] = "Random Forest Classifier"
        self.full_name["linsvm"] = "Linear SVM"
        self.full_name["gdsvm"] = "Gradient Descent SVM"
        self.full_name["gdlr"] = "Gradient Descent Logistic Regression"
        self.full_name["lr"] = "Logistic Regression"
        self.full_name["mnb"] = "Multinomial Naive Bayes"
        self.full_name["cnb"] = "Complement Naive Bayes"
        self.full_name["bnb"] = "Bernoulli Naive Bayes"
        self.full_name["dectree"] = "Decision Tree"
        self.full_name["neuralnet"] = "MLP Classifier (Neural Net)"
        return self.full_name

    def get_hyper_params(self):
        self.hyper_params["ridgeclf"] = {'alpha': [0.01,1.0]}
        self.hyper_params["pcpt"] = {'penalty': ["l2","l1","elasticnet", None], 'alpha': [0.0001, 0.0005, 0.001, 0.01, 0.1, 1.0]}
        self.hyper_params["knn"] = {'n_neighbors': [5,10,15], 'leaf_size': [20,30,40], 'weights':['uniform', 'distance']}
        self.hyper_params["rfc"] = {'max_depth': [2,3,4,5, None], 'ccp_alpha':[0.0, 0.01, 0.1, 0.5], 'oob_score': [True, False]}
        self.hyper_params["linsvm"] = {'C':[0.5, 1,2,5,10,100, 500, 1000], 'loss': ["hinge", "squared_hinge"], 'penalty':['l1', 'l2']}
        self.hyper_params["gdsvm"] = {'alpha': [0.0001,0.0005, 0.001, 0.01, 0.1, 1]}
        self.hyper_params["gdlr"] = {'alpha': [0.0001,0.0005, 0.001, 0.01, 0.1, 1], 'penalty': ['l2', 'l1', 'elasticnet']}
        self.hyper_params["lr"] = {'C':[0.5, 1.0,2,5,10,100], 'penalty': ['l2','l1','elasticnet', 'none']}
        self.hyper_params["mnb"] = {'alpha' : [0.01, 0.05, 0.1, 1.0, 2.0]}
        self.hyper_params["cnb"] = {'alpha' : [0.01, 0.05, 0.1, 1.0, 2.0]}
        self.hyper_params["bnb"] = {'alpha' : [0.01, 0.05, 0.1, 1.0, 2.0]}
        self.hyper_params["dectree"] = {'max_depth': [2,3,4,5, None], 'ccp_alpha':[0.0, 0.01, 0.1, 0.5]}
        self.hyper_params["neuralnet"] = {}
        return self.hyper_params

    def print_metrics(self, gs):
        for metric in self.scoring_metrics:
            means = gs.cv_results_['mean_test_' + metric]
            stds = gs.cv_results_['std_test_' + metric]
            best_param = gs.best_params_
            for mean, std, params in zip(means, stds, gs.cv_results_['params']):
                if params == best_param:
                    print(metric+":","%0.3f (+/-%0.03f) for %r"
                          % (mean, std * 2, params))

    def process_classifier(self, clf, name, hyperparams, X_train, y_train, verbose = False):
        start = time.time()
        print("-"*10, "Processing {}".format(self.full_name[name]), "-"*10)
        gs = GridSearchCV(clf, param_grid = hyperparams[name],
                          scoring = self.scoring_metrics, cv = 5, refit = self.refit)
        gs.fit(X_train, y_train)
        duration = time.time() - start
        if duration > 60:
            print(self.full_name[name], "completed in {:.2f} minutes.".format(duration/60),"\n\n")
        else:
            print(self.full_name[name], "completed in {:.2f} seconds.".format(duration),"\n\n")

        if verbose:
            print("Grid Search Results:", name)
            print(gs.best_estimator_)
            self.print_metrics(gs)
            print("\n")

        return gs.best_estimator_, name

    def plot_learning_curve(self, name, train_sizes, train_scores, test_scores):
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            # Plot learning curve
            plt.title(name)
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1,
                                 color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1,
                                 color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                         label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                         label="Cross-validation score")
            plt.legend(loc="best")

            plt.show()


    def study_tuned_classifier(self, clf, name, X, y):
        train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=5)
        plot_learning_curve(name, train_sizes, train_scores, test_scores)
        return([name, train_sizes, train_scores, test_scores])

    def evaluate_clf_on_heldout_data(self, clf, name, X_test, y_test):
        print("-"*10, "Evaluate on held-out data, {}".format(self.full_name[name]), "-"*10)
        pred = clf.predict(X_test)
        print(classification_report(y_test, pred, target_names=self.target_names))
        if hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(X_test)
        elif hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(X_test)
        
        roc_auc = roc_auc_score(y_test, y_score, average='weighted')
        print("ROC AUC Score:\t\t", roc_auc,'\n\n')
        return roc_auc

    def cross_validate_clf(self, clf, name, X, y, verbose = False):
        print("-"*10, "Evaluate on Cross-Val, {}".format(name), "-"*10)
        accuracies = []
        rocs = []
        for train_index, test_index in self.skf.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf_copy = clone(clf)
            clf_copy.fit(X_train,y_train)
            pred = clf_copy.predict(X_test)
            accuracies.append(accuracy_score(pred, y_test))
            if hasattr(clf_copy, 'decision_function'):
                y_score = clf_copy.decision_function(X_test)
                rocs.append(roc_auc_score(y_test, y_score, average='weighted'))
        if verbose:
            print("raw accuracies:", accuracies)
            print("raw rocs:", rocs)
        print("accuracy:", (sum(accuracies) / 5), "\t", "roc:",(sum(rocs)/5))
        return (sum(accuracies) / 5), (sum(rocs)/5)



    def perform_training(self, X, y, classifiers, verbose = False):
        y_full_score = {}
        y_full_pred = {}
        y_full_true = {}

        for _, name in classifiers : 
                y_full_pred[name] = {}
                y_full_score[name] = {}

        split_iter = 0
        for train_index, test_index in self.skf.split(X, y):
            split_iter+=1
            print("Iteration Number: " + str(split_iter) + "\n\n")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print(pd.Series(y_train).value_counts())
            print("Pretuning")

            gdsvm_clf = [classifier for classifier in classifiers if classifier[1] == 'gdsvm'][0]
            gdsvm_tuned = self.process_classifier(gdsvm_clf[0], gdsvm_clf[1], self.hyper_params, X_train, y_train, verbose) # pre_tuned_classifiers[3][0]

            select_from_model = SelectFromModel(gdsvm_tuned[0], prefit=True)
            
            X_train_new = select_from_model.transform(X_train)
            X_test_new = select_from_model.transform(X_test)
            X_new = select_from_model.transform(X)

            print("Tuning")
            tuned_classifiers = [self.process_classifier(clf, name, self.hyper_params, X_train_new, y_train, verbose) for clf, name in classifiers]

            y_full_true[split_iter] = y_test


            for clf,name  in tuned_classifiers:

                y_pred=clf.predict(X_test_new) 
                print(self.evaluate_clf_on_heldout_data(clf, name, X_test_new, y_test))

                if hasattr(clf, 'predict_proba'):
                    y_score = clf.predict_proba(X_test_new)[:,1]
                elif hasattr(clf, 'decision_function'):
                    y_score = clf.decision_function(X_test_new)
                else:
                    print("No function", name)
                # pickle y_score and clf.predict(x_train) for every clf and config to plot results later?
                y_full_pred[name][split_iter] = y_pred
                y_full_score[name][split_iter] = y_score

        return [y_full_score, y_full_pred, y_full_true]
        #pickle.dump(to_pickle, open('../results_pickled/' + column_name + '_relevancy_2807.pb', 'wb'))

    #     rocaucs = plot_roc_data(y_full_score, y_full_pred, y_full_true, column_name)
    #     avgprs = plot_pr_data(y_full_score, y_full_pred, y_full_true, column_name)
    #     return process_results(rocaucs, avgprs, convergence_metric)
    
    
    def fit_final_classifier(self, clf, X, y):
        print("Pretuning")
        pretuned = self.process_classifier(clf[0], clf[1], self.hyper_params, X, y) # pre_tuned_classifiers[3][0]

        model = SelectFromModel(pretuned[0], prefit=True)
        X_new = model.transform(X)
        print("Tuning")
        tuned = self.process_classifier(pretuned[0], pretuned[1], self.hyper_params, X_new, y)
        return tuned, model
   
    
    def calculate_scores(self, y_scores, y_preds, y_true):
        for model in y_scores.keys():
            print(model)
            rocauc = []
            prauc = []
            f1s = []
            for fold_id in range(1,6):
                rocauc.append(roc_auc_score(y_true[fold_id], y_scores[model][fold_id])) 
                prauc.append(average_precision_score(y_true[fold_id], y_scores[model][fold_id]))
                f1s.append(f1_score(y_true[fold_id], y_preds[model][fold_id], average='weighted'))
                
            np_rocauc = np.array(rocauc)
            np_prauc = np.array(prauc)
            np_f1s = np.array(f1s)
            
            roc_mean = round(np.mean(np_rocauc, axis = 0), 3)
            pr_mean = round(np.mean(np_prauc, axis=0), 3)
            f1_mean = round(np.mean(np_f1s, axis=0), 3)
            
            roc_ci_std =round( 1.960 * (np.std(np_rocauc, axis=0) / 2.236), 3)
            pr_ci_std = round( 1.960 * (np.std(np_rocauc, axis=0) / 2.236), 3)
            f1_ci_std = round( 1.960 * (np.std(np_f1s, axis=0) / 2.236), 3)
            
            print("Average PR AUC: ", pr_mean, pr_mean - pr_ci_std, pr_mean + pr_ci_std)
            print("Average ROC AUC: ", roc_mean, roc_mean - roc_ci_std, roc_mean + roc_ci_std )
            print("Average F1 Weighted: ", f1_mean, f1_mean - f1_ci_std, f1_mean + roc_ci_std )
            print("\n")
