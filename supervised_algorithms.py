import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class SupervisedAlgorithms:
    """Class to operate with a dataset in CSV format"""

    def __init__(self):
        self.fig_width = 20
        self.fig_height = 10
        self.bar_width = 0.25
        self.X_train = None
        self.X_train_scaled = None
        self.X_test = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

    def train_test_split(self, feature_data, class_data, test_size, algorithm):
        """Split data into training and test datasets and plot the class distribution in each set"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(feature_data, class_data,
                                                                                test_size=test_size, shuffle=True,
                                                                                stratify=class_data, random_state=1)
        self.y_train = np.ravel(self.y_train)
        self.y_test = np.ravel(self.y_test)
        print("X_train type: {} and shape: {}".format(type(self.X_train), self.X_train.shape))
        print("X_test type: {} and shape: {}".format(type(self.X_test), self.X_test.shape))
        print("y_train type: {} and shape: {}".format(type(self.y_train), self.y_train.shape))
        print("y_test type: {} and shape: {} \n".format(type(self.y_test), self.y_test.shape))
        self.data_scaling(algorithm)
        return self.X_train_scaled, self.y_train, self.X_test_scaled, self.y_test

    def data_scaling(self, algorithm):
        """Scaling data to normalization or standardization"""
        if algorithm.lower() == 'norm':
            scaler = MinMaxScaler()
            scaler.fit(self.X_train)
            self.X_train_scaled = scaler.transform(self.X_train)
            self.X_test_scaled = scaler.transform(self.X_test)
        elif algorithm.lower() == 'std':
            scaler = StandardScaler()
            scaler.fit(self.X_train)
            self.X_train_scaled = scaler.transform(self.X_train)
            self.X_test_scaled = scaler.transform(self.X_test)
        else:
            print('Data NOT scaled, algorithm is not correct')
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test
        print("X_train_scaled type: {} and shape: {}".format(type(self.X_train_scaled), self.X_train_scaled.shape))
        print("X_test_scaled type: {} and shape: {} \n".format(type(self.X_test_scaled), self.X_test_scaled.shape))

    def apply_algorithm(self, alg, pars, results, algorithm, params):
        """Apply the machine learning algorithm to the train and test datasets"""
        time0 = time.time()
        if algorithm.lower() == 'knn':
            model = KNeighborsClassifier()
        elif algorithm.lower() == 'logreg':
            model = LogisticRegression(random_state=0)
        elif algorithm.lower() == 'linearsvc':
            model = LinearSVC(random_state=0)
        elif algorithm.lower() == 'naivebayes':
            model = GaussianNB()
        elif algorithm.lower() == 'tree':
            model = DecisionTreeClassifier(random_state=0)
        elif algorithm.lower() == 'forest' or algorithm.lower() == 'random':
            model = RandomForestClassifier(random_state=0)
        elif algorithm.lower() == 'gradient':
            model = GradientBoostingClassifier(random_state=0)
        elif algorithm.lower() == 'svm':
            model = SVC(random_state=0)
        elif algorithm.lower() == 'mlp':
            model = MLPClassifier(random_state=0)
        else:
            print('Algorithm was NOT provided.')
            return None
        for key, value in params.items():
            setattr(model, key, value)
        print('SCORE WITH {} ALGORITHM AND PARAMS {}\n'.format(algorithm, params))
        model.fit(self.X_train, self.y_train)
        time1 = time.time()
        unscaled_model_time = round(time1 - time0, 4)
        unscaled_train_score = round(model.score(self.X_train, self.y_train), 4)
        unscaled_test_score = round(model.score(self.X_test, self.y_test), 4)
        print('Unscaled modeling time [seconds]: {}'.format(unscaled_model_time, 4))
        print('Unscaled TRAIN dataset: {}'.format(unscaled_train_score, 4))
        print('Unscaled TEST dataset: {}'.format(unscaled_test_score, 4))
        time2 = time.time()
        unscaled_predict_time = round(time2 - time1, 4)
        print('Unscaled predicting time [seconds]: {}\n'.format(unscaled_predict_time, 4))
        model.fit(self.X_train_scaled, self.y_train)
        time3 = time.time()
        scaled_model_time = round(time3 - time2, 4)
        scaled_train_score = round(model.score(self.X_train_scaled, self.y_train), 4)
        scaled_test_score = round(model.score(self.X_test_scaled, self.y_test), 4)
        print('Scaled modeling time [seconds]: {}'.format(scaled_model_time, 4))
        print('Scaling TRAIN dataset: {}'.format(scaled_train_score, 4))
        print('Scaling TEST dataset: {}'.format(scaled_test_score, 4))
        time4 = time.time()
        scaled_predict_time = round(time4 - time3, 4)
        print('Scaled predicting time [seconds]: {}\n\n'.format(scaled_predict_time, 4))
        out = np.array([[unscaled_model_time, unscaled_predict_time, unscaled_train_score, unscaled_test_score,
                         scaled_model_time, scaled_predict_time, scaled_train_score, scaled_test_score]])
        alg.append(algorithm)
        pars.append(params)
        if results.shape[0] == 0:
            results = np.zeros([0, out.shape[1]])
        results = np.append(results, out, axis=0)
        return alg, pars, results

    def cross_grid_validation(self, algorithm, scale, param_grid, nfolds=5):
        time0 = time.time()
        model = []
        scaler = []
        for i in range(len(algorithm)):
            if algorithm[i].lower() == 'knn':
                model.append(KNeighborsClassifier())
            elif algorithm[i].lower() == 'logreg':
                model.append(LogisticRegression(random_state=0))
            elif algorithm[i].lower() == 'linearsvc':
                model.append(LinearSVC(random_state=0))
            elif algorithm[i].lower() == 'naivebayes':
                model.append(GaussianNB())
            elif algorithm[i].lower() == 'tree':
                model.append(DecisionTreeClassifier(random_state=0))
            elif algorithm[i].lower() == 'forest' or algorithm[i].lower() == 'random':
                model.append(RandomForestClassifier(random_state=0))
            elif algorithm[i].lower() == 'gradient':
                model.append(GradientBoostingClassifier(random_state=0))
            elif algorithm[i].lower() == 'svm':
                model.append(SVC(random_state=0))
            elif algorithm[i].lower() == 'mlp':
                model.append(MLPClassifier(random_state=0))
            else:
                print('Algorithm was NOT provided. Note the type must be a list.')
                return None
            if scale[i].lower() == 'norm':
                scaler.append(MinMaxScaler())
            elif scale[i].lower() == 'std':
                scaler.append(StandardScaler())
            else:
                scaler.append(None)
            param_grid[i]['classifier'] = [model[i]]
            param_grid[i]['preprocessing'] = [scaler[i]]
        pipe = Pipeline([('preprocessing', scaler), ('classifier', model)])
        grid_search = GridSearchCV(pipe, param_grid, cv=nfolds, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        print("Best parameters: {}".format(grid_search.best_params_))
        print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))
        print("Test set score: {:.4f}".format(grid_search.score(self.X_test, self.y_test)))
        print('Grid search time: {:.1f}'.format(time.time() - time0))
        return grid_search
