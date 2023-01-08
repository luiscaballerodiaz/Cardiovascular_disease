import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class DataBinaryOutputCSV:
    """Class to operate with a dataset in CSV format"""

    def __init__(self, name):
        self.name = name
        self.fig_width = 20
        self.fig_height = 10
        self.bar_width = 0.25
        self.percentile = 0.02
        self.pca = None
        self.X_train = None
        self.X_train_scaled = None
        self.X_train_scaled_pca = None
        self.X_train_scaled_pca_output0 = None
        self.X_train_scaled_pca_output1 = None
        self.X_test = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.y_train_output0 = None
        self.y_train_output1 = None
        self.y_test_output0 = None
        self.y_test_output1 = None

    def read_csv(self):
        dataset = pd.read_csv(self.name)
        print("Full source data from CSV type: {} and shape: {}".format(type(dataset), dataset.shape))
        return dataset

    @staticmethod
    def binary_output_split(dataset, class_column_name):
        output0 = dataset.loc[dataset[class_column_name] == 0, :]
        output1 = dataset.loc[dataset[class_column_name] == 1, :]
        print("Cases class = 0 type: {} and shape: {}".format(type(output0), output0.shape))
        print("Cases class = 1 type: {} and shape: {} \n".format(type(output1), output1.shape))
        return [output0, output1]

    def binary_class_histogram(self, dataset, class_column_name, plot_name, x_axes_name, y_axes_name, plot_legend):
        [output0, output1] = self.binary_output_split(dataset, class_column_name)
        fig, axes = plt.subplots(math.ceil(dataset.shape[1] / 2), 2, figsize=(self.fig_width, self.fig_height))
        if dataset.shape[1] % 2 == 1:
            fig.delaxes(axes[math.ceil(dataset.shape[1] / 2) - 1, 1])
        ax = axes.ravel()
        for i in range(dataset.shape[1]):
            ax[i].hist(output0.iloc[:, i], histtype='stepfilled', bins=25, alpha=0.25, color="#0000FF", lw=0)
            ax[i].hist(output1.iloc[:, i], histtype='stepfilled', bins=25, alpha=0.25, color='#FF0000', lw=0)
            ax[i].set_title(dataset.keys()[i], fontsize=10, fontweight='bold')
            ax[i].grid(visible=True)
            ax[i].tick_params(axis='both', labelsize=8)
            ax[i].set_ylabel(y_axes_name, fontsize=8)
            ax[i].set_xlabel(x_axes_name, fontsize=8)
        ax[0].legend(plot_legend, loc="best")
        plt.savefig(plot_name, bbox_inches='tight')
        plt.clf()

    def data_scrubbing(self, dataset, columns_to_remove, concept1, concept2):
        # Remove non-meaningful columns
        dataset.drop(columns_to_remove, axis=1, inplace=True)
        print("Scrubber data after eliminating non-meaningful columns type: {} and shape: {}".format(type(dataset),
                                                                                                     dataset.shape))
        # Remove duplicates
        dataset.drop_duplicates(keep='first', inplace=True)
        print("Scrubber data after eliminating duplicates type: {} and shape: {}".format(type(dataset), dataset.shape))
        # Remove outliers
        df_qmin = dataset.quantile(self.percentile)
        df_qmax = dataset.quantile(1 - self.percentile)
        for i in range(len(dataset.keys())):
            if min(dataset.iloc[:, i]) == 0 and max(dataset.iloc[:, i] <= 3):
                continue
            else:
                dataset = dataset.loc[dataset[dataset.keys()[i]] >= df_qmin[i], :]
                dataset = dataset.loc[dataset[dataset.keys()[i]] <= df_qmax[i], :]
        print("Scrubber data after eliminating outliers type: {} and shape: {}".format(type(dataset), dataset.shape))
        # Remove empty rows
        dataset.replace('', np.nan, inplace=True)
        dataset.dropna(axis=0, how='any', inplace=True)
        dataset.reset_index(drop=True, inplace=True)
        print("Scrubber data after eliminating empty datasets type: {} and shape: {}".format(type(dataset),
                                                                                             dataset.shape))
        # Remove wrong rows if concept1 is higher than concept2
        index_to_drop = []
        for i in range(dataset.shape[0]):
            if dataset.iloc[i, dataset.columns.get_loc(concept1)] > dataset.iloc[i, dataset.columns.get_loc(concept2)]:
                index_to_drop.append(i)
        dataset.drop(index_to_drop, inplace=True)
        dataset.reset_index(drop=True, inplace=True)
        print("Scrubber data after eliminating non-consistent datasets type: {} and shape: {}".format(type(dataset),
                                                                                                      dataset.shape))
        return dataset

    def train_test_split(self, feature_data, class_data, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(feature_data, class_data,
                                                                                test_size=test_size, shuffle=True,
                                                                                stratify=class_data, random_state=1)
        self.y_train = np.ravel(self.y_train)
        self.y_test = np.ravel(self.y_test)
        print("X_train type: {} and shape: {}".format(type(self.X_train), self.X_train.shape))
        print("X_test type: {} and shape: {}".format(type(self.X_test), self.X_test.shape))
        print("y_train type: {} and shape: {}".format(type(self.y_train), self.y_train.shape))
        print("y_test type: {} and shape: {} \n".format(type(self.y_test), self.y_test.shape))
        self.plot_output_class_distribution()

    def plot_output_class_distribution(self):
        self.y_train_output1 = self.y_train[self.y_train == 1]
        self.y_train_output0 = self.y_train[self.y_train == 0]
        self.y_test_output1 = self.y_test[self.y_test == 1]
        self.y_test_output0 = self.y_test[self.y_test == 0]
        print("y_train_output1 type: {} and shape: {}".format(type(self.y_train_output1), self.y_train_output1.shape))
        print("y_test_output1 type: {} and shape: {}".format(type(self.y_test_output1), self.y_test_output1.shape))
        print("y_train_output0 type: {} and shape: {}".format(type(self.y_train_output0), self.y_train_output0.shape))
        print("y_test_output0 type: {} and shape: {} \n".format(type(self.y_test_output0), self.y_test_output0.shape))

        plt.subplots(figsize=(self.fig_width, self.fig_height))
        plt.bar([1, 2], [self.y_train_output0.shape[0], self.y_test_output0.shape[0]],
                color='r', width=self.bar_width, edgecolor='black', label='class=0')
        plt.bar([1 + self.bar_width, 2 + self.bar_width], [self.y_train_output1.shape[0], self.y_test_output1.shape[0]],
                color='b', width=self.bar_width, edgecolor='black', label='class=1')
        plt.xticks([1 + self.bar_width / 2, 2 + self.bar_width / 2],
                   ['Train data', 'Test data'], ha='center')
        plt.text(1 - self.bar_width / 4, self.y_train_output0.shape[0] + 100,
                 str(self.y_train_output0.shape[0]), fontsize=20)
        plt.text(1 + 3 * self.bar_width / 4, self.y_train_output1.shape[0] + 100,
                 str(self.y_train_output1.shape[0]), fontsize=20)
        plt.text(2 - self.bar_width / 4, self.y_test_output0.shape[0] + 100,
                 str(self.y_test_output0.shape[0]), fontsize=20)
        plt.text(2 + 3 * self.bar_width / 4, self.y_test_output1.shape[0] + 100,
                 str(self.y_test_output1.shape[0]), fontsize=20)
        plt.title('Output class distribution between train and test datasets', fontsize=24)
        plt.xlabel('Concepts', fontweight='bold', fontsize=14)
        plt.ylabel('Count train/test class cases', fontweight='bold', fontsize=14)
        plt.legend()
        plt.grid()
        plt.savefig('Count_class_cases.png', bbox_inches='tight')
        plt.clf()

    def data_scaling(self, algorithm):
        if algorithm.lower() == 'norm':
            scaler = MinMaxScaler()
        elif algorithm.lower() == 'standard':
            scaler = StandardScaler()
        else:
            print('Algorithm not correct')
            return None
        scaler.fit(self.X_train)
        self.X_train_scaled = scaler.transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        print("X_train_scaled type: {} and shape: {}".format(type(self.X_train_scaled), self.X_train_scaled.shape))
        print("X_test_scaled type: {} and shape: {} \n".format(type(self.X_test_scaled), self.X_test_scaled.shape))

    def apply_pca(self, ncomps):
        self.pca = PCA(n_components=ncomps)
        self.pca.fit(self.X_train_scaled)
        self.X_train_scaled_pca = self.pca.transform(self.X_train_scaled)
        print("X_train_scaled PCA type: {} and shape: {}".format(type(self.X_train_scaled_pca),
                                                                 self.X_train_scaled_pca.shape))
        self.X_train_scaled_pca_output1 = self.X_train_scaled_pca[self.y_train == 1, :]
        self.X_train_scaled_pca_output0 = self.X_train_scaled_pca[self.y_train == 0, :]
        print("X_train_scaled PCA output = 1 type: {} and shape: {}"
              .format(type(self.X_train_scaled_pca_output1), self.X_train_scaled_pca_output1.shape))
        print("X_train_scaled PCA output = 0 type: {} and shape: {}"
              .format(type(self.X_train_scaled_pca_output0), self.X_train_scaled_pca_output0.shape))
        print("PCA component shape: {} \n".format(self.pca.components_.shape))
        self.plot_pca_breakdown()
        self.plot_pca_scree()
        if ncomps > 2:
            self.plot_first_second_pca()

    def plot_pca_breakdown(self):
        _, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        plt.pcolormesh(self.pca.components_, cmap=plt.cm.cool)
        plt.colorbar()
        pca_range = [x + 0.5 for x in range(self.pca.components_.shape[1])]
        plt.xticks(pca_range, self.X_train.keys(), rotation=60, ha='center')
        ax.xaxis.tick_top()
        str_pca = []
        for i in range(self.X_train.shape[1]):
            str_pca.append('Component ' + str(i + 1))
        plt.yticks(pca_range, str_pca)
        plt.xlabel("Feature", weight='bold', fontsize=14)
        plt.ylabel("Principal components", weight='bold', fontsize=14)
        plt.savefig('PCA_scaled_breakdown.png', bbox_inches='tight')
        plt.clf()

    def plot_pca_scree(self):
        fig, ax1 = plt.subplots(figsize=(self.fig_width, self.fig_height))
        ax2 = ax1.twinx()
        label1 = ax1.plot(range(1, len(self.pca.components_) + 1), self.pca.explained_variance_ratio_,
                          'ro-', linewidth=2, label='Individual PCA variance')
        label2 = ax2.plot(range(1, len(self.pca.components_) + 1), np.cumsum(self.pca.explained_variance_ratio_),
                          'b^-', linewidth=2, label='Cumulative PCA variance')
        plt.title('Scree Plot', fontsize=20, fontweight='bold')
        ax1.set_xlabel('Principal Components', fontsize=14)
        ax1.set_ylabel('Proportion of Variance Explained', fontsize=14, color='r')
        ax2.set_ylabel('Cumulative Proportion of Variance Explained', fontsize=14, color='b')
        la = label1 + label2
        lb = [la[0].get_label(), la[1].get_label()]
        ax1.legend(la, lb, loc='upper center')
        ax1.grid(visible=True)
        ax2.grid(visible=True)
        plt.savefig('PCA_scaled_ScreePlot.png', bbox_inches='tight')
        plt.clf()

    def plot_first_second_pca(self):
        plt.subplots(figsize=(self.fig_width, self.fig_height))
        plt.scatter(self.X_train_scaled_pca_output1[:, 0], self.X_train_scaled_pca_output1[:, 1],
                    s=10, marker='^', c='red', label='output=1')
        plt.scatter(self.X_train_scaled_pca_output0[:, 0], self.X_train_scaled_pca_output0[:, 1],
                    s=10, marker='o', c='blue', label='output=0')
        plt.title('Cardiovascular disease modelling', fontsize=20, fontweight='bold')
        plt.xlabel('First PCA', fontsize=14)
        plt.ylabel('Second PCA', fontsize=14)
        plt.legend()
        plt.grid()
        plt.savefig('PCA_scaled_First_Second.png', bbox_inches='tight')
        plt.clf()

    def apply_algorithm(self, algorithm, params):
        time0 = time.time()
        if algorithm.lower() == 'knn':
            model = KNeighborsClassifier(n_neighbors=params['n_neighbors'], weights=params['weights'])
        elif algorithm.lower() == 'logreg':
            model = LogisticRegression(random_state=params['random_state'], C=params['C'], max_iter=params['max_iter'])
        elif algorithm.lower() == 'linearsvc':
            model = LinearSVC(random_state=params['random_state'], C=params['C'], max_iter=params['max_iter'])
        elif algorithm.lower() == 'naivebayes':
            model = GaussianNB()
        elif algorithm.lower() == 'tree':
            model = DecisionTreeClassifier(random_state=params['random_state'], max_depth=params['max_depth'])
        elif algorithm.lower() == 'forest':
            model = RandomForestClassifier(random_state=params['random_state'], max_depth=params['max_depth'],
                                           n_estimators=params['n_estimators'], max_features=params['max_features'])
        elif algorithm.lower() == 'gradient':
            model = GradientBoostingClassifier(random_state=params['random_state'], max_depth=params['max_depth'],
                                               learning_rate=params['learning_rate'],
                                               n_estimators=params['n_estimators'])
        elif algorithm.lower() == 'svm':
            model = SVC(random_state=params['random_state'], kernel=params['kernel'], C=params['C'],
                        gamma=params['gamma'])
        elif algorithm.lower() == 'mlp':
            model = MLPClassifier(random_state=params['random_state'], activation=params['activation'],
                                  hidden_layer_sizes=params['hidden_layer_sizes'], alpha=params['alpha'])
        else:
            return None
        print('SCORE WITH {} ALGORITHM AND PARAMS {}\n'.format(algorithm, params))
        model.fit(self.X_train, self.y_train)
        time1 = time.time()
        print('Unscaled modeling time [seconds]: {}'.format(str(time1 - time0)))
        print('Unscaled TRAIN dataset: {}'.format(str(model.score(self.X_train, self.y_train))))
        print('Unscaled TEST dataset: {}'.format(str(model.score(self.X_test, self.y_test))))
        time2 = time.time()
        print('Unscaled predicting time [seconds]: {}\n'.format(str(time2 - time1)))
        model.fit(self.X_train_scaled, self.y_train)
        time3 = time.time()
        print('Scaled modeling time [seconds]: {}'.format(str(time3 - time2)))
        print('Scaling TRAIN dataset: {}'.format(str(model.score(self.X_train_scaled, self.y_train))))
        print('Scaling TEST dataset: {}'.format(str(model.score(self.X_test_scaled, self.y_test))))
        time4 = time.time()
        print('Scaled predicting time [seconds]: {}\n\n'.format(str(time4 - time3)))
