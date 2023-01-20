import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import numpy as np


class DataPlot:
    """Class to plot data using visualization tools"""

    def __init__(self):
        self.fig_width = 20
        self.fig_height = 10
        self.bar_width = 0.25
        self.nfeat = 2

    @staticmethod
    def binary_output_split(dataset, class_column_name):
        """"Split the input dataset according to the binary class value """
        output0 = dataset.loc[dataset[class_column_name] == 0, :]
        output1 = dataset.loc[dataset[class_column_name] == 1, :]
        print("Cases class = 0 type: {} and shape: {}".format(type(output0), output0.shape))
        print("Cases class = 1 type: {} and shape: {} \n".format(type(output1), output1.shape))
        return [output0, output1]

    def boxplot(self, dataset, plot_name, max_features_row):
        """Plot boxplot based on input dataset"""
        dfcopy = dataset.copy()
        max_vector = np.zeros([dataset.shape[1]])
        for i in range(dataset.shape[1]):
            max_vector[i] = dataset.iloc[:, i].max()
        columns = []
        for i in range(dataset.shape[1]):
            index_max = np.argmax(max_vector)
            columns.append(dataset.columns.values[index_max])
            max_vector[index_max] = 0
        dfcopy = dfcopy.reindex(columns=columns)
        dfcopy.replace(np.nan, 0, inplace=True)
        fig, axes = plt.subplots(math.ceil(dataset.shape[1] / max_features_row), 1,
                                 figsize=(self.fig_width, self.fig_height))
        ax = axes.ravel()
        for i in range(len(ax)):
            ax[i].boxplot(dfcopy.iloc[:, (i * max_features_row):min(((i + 1) * max_features_row), dataset.shape[1])])
            ax[i].grid(visible=True)
            ax[i].tick_params(axis='both', labelsize=8)
            if ((i + 1) * max_features_row) > dataset.shape[1]:
                xrange = range(1, dataset.shape[1] - (i * max_features_row) + 1)
            else:
                xrange = range(1, max_features_row + 1)
            ax[i].set_xticks(xrange,
                             dfcopy.keys()[(i * max_features_row):min(((i + 1) * max_features_row), dataset.shape[1])],
                             rotation=10, ha='center')
            ax[i].set_ylabel('Feature magnitude', fontsize=8)
        ax[0].set_title(plot_name, fontsize=24, fontweight='bold')
        plt.subplots_adjust(top=0.85)
        fig.tight_layout(h_pad=2)
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.clf()

    def binary_class_histogram(self, dataset, class_column_name, plot_name, ncolumns):
        """Plot histogram based on input dataset"""
        [output0, output1] = self.binary_output_split(dataset, class_column_name)
        fig, axes = plt.subplots(math.ceil(dataset.shape[1] / ncolumns), ncolumns,
                                 figsize=(self.fig_width, self.fig_height))
        spare_axes = ncolumns - dataset.shape[1] % ncolumns
        if spare_axes == ncolumns:
            spare_axes = 0
        for axis in range(ncolumns - 1, ncolumns - 1 - spare_axes, -1):
            fig.delaxes(axes[math.ceil(dataset.shape[1] / ncolumns) - 1, axis])
        ax = axes.ravel()
        for i in range(dataset.shape[1]):
            ax[i].hist(output0.iloc[:, i], histtype='stepfilled', bins=25, alpha=0.25, color="#0000FF", lw=0)
            ax[i].hist(output1.iloc[:, i], histtype='stepfilled', bins=25, alpha=0.25, color='#FF0000', lw=0)
            ax[i].set_title(dataset.keys()[i], fontsize=10, y=1.0, pad=-14, fontweight='bold')
            ax[i].grid(visible=True)
            ax[i].tick_params(axis='both', labelsize=8)
            ax[i].set_ylabel('Frequency', fontsize=8)
            ax[i].set_xlabel('Feature magnitude', fontsize=8)
        ax[0].legend(['output0', 'output1'], loc="best")
        plt.subplots_adjust(top=0.85)
        fig.tight_layout(h_pad=2)
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.clf()

    def plot_output_class_distribution(self, y_train, y_test):
        """Plot the class distribution in the training and test dataset"""
        y_train_output1 = y_train[y_train == 1]
        y_train_output0 = y_train[y_train == 0]
        y_test_output1 = y_test[y_test == 1]
        y_test_output0 = y_test[y_test == 0]
        print("y_train_output1 type: {} and shape: {}".format(type(y_train_output1), y_train_output1.shape))
        print("y_test_output1 type: {} and shape: {}".format(type(y_test_output1), y_test_output1.shape))
        print("y_train_output0 type: {} and shape: {}".format(type(y_train_output0), y_train_output0.shape))
        print("y_test_output0 type: {} and shape: {} \n".format(type(y_test_output0), y_test_output0.shape))

        plt.subplots(figsize=(self.fig_width, self.fig_height))
        plt.bar([1, 2], [y_train_output0.shape[0], y_test_output0.shape[0]],
                color='r', width=self.bar_width, edgecolor='black', label='class=0')
        plt.bar([1 + self.bar_width, 2 + self.bar_width], [y_train_output1.shape[0], y_test_output1.shape[0]],
                color='b', width=self.bar_width, edgecolor='black', label='class=1')
        plt.xticks([1 + self.bar_width / 2, 2 + self.bar_width / 2],
                   ['Train data', 'Test data'], ha='center')
        plt.text(1 - self.bar_width / 4, y_train_output0.shape[0] + 100,
                 str(y_train_output0.shape[0]), fontsize=20)
        plt.text(1 + 3 * self.bar_width / 4, y_train_output1.shape[0] + 100,
                 str(y_train_output1.shape[0]), fontsize=20)
        plt.text(2 - self.bar_width / 4, y_test_output0.shape[0] + 100,
                 str(y_test_output0.shape[0]), fontsize=20)
        plt.text(2 + 3 * self.bar_width / 4, y_test_output1.shape[0] + 100,
                 str(y_test_output1.shape[0]), fontsize=20)
        plt.title('Output class distribution between train and test datasets', fontsize=24)
        plt.xlabel('Concepts', fontweight='bold', fontsize=14)
        plt.ylabel('Count train/test class cases', fontweight='bold', fontsize=14)
        plt.legend()
        plt.grid()
        plt.savefig('Count_class_cases.png', bbox_inches='tight')
        plt.clf()

    def param_sweep_plot(self, algorithm, params, test_score):
        for i in range(len(algorithm)):
            test = []
            feat_name = []
            feat1 = []
            feat2 = []
            feat3 = []
            if algorithm[i] == 'linearsvc' or algorithm[i] == 'linear svc':
                algorithm[i] = 'LinearSVC'
            if algorithm[i] == 'logreg' or algorithm[i] == 'logistic regression':
                algorithm[i] = 'LogisticRegression'
            if algorithm[i].lower() == 'svm':
                algorithm[i] = 'SVC'
            if 'naive' in algorithm[i].lower() or 'bayes' in algorithm[i].lower():
                algorithm[i] = 'NB'
            for j in range(len(params)):
                string = str(params[j]['classifier'])
                if (algorithm[i].title() in string) or (algorithm[i].upper() in string):
                    test.append(test_score[j])
                    for key, value in params[j].items():
                        if 'classifier__' in key:
                            key = key.replace('classifier__', '')
                            if key not in feat_name:
                                feat_name.append(key)
                            for k in range(len(feat_name)):
                                if feat_name[k] == key:
                                    if k == 0:
                                        feat1.append(value)
                                    elif k == 1:
                                        feat2.append(value)
                                    elif k == 2:
                                        feat3.append(value)
            if not feat_name:
                fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
                ax.set_title('Test score assessment per parameter sweep with ' + algorithm[i].upper() + ' algorithm',
                             fontsize=24)
                ax.text(0.5, 0.5, str(round(test[0], 4)), ha="center", va="center", color="k", fontweight='bold',
                        fontsize=10)
                fig.tight_layout(h_pad=2)
            if len(feat_name) == 1:
                test_matrix = np.array([test])
                fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
                plt.pcolormesh(test_matrix, cmap=plt.cm.PuBuGn)
                plt.colorbar()
                ax.set_xlabel('Parameter sweep ' + feat_name[0], fontsize=14)
                ax.set_title('Test score assessment per parameter sweep with ' + algorithm[i].upper() + ' algorithm',
                             fontsize=24)
                ax.set_xticks(np.arange(0.5, len(feat1) + 0.5), labels=feat1, fontsize=14)
                plt.yticks([])
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                for j in range(len(feat1)):
                    ax.text(j + 0.5, 0.5, str(round(test_matrix[0, j], 4)),
                            ha="center", va="center", color="k", fontweight='bold', fontsize=10)
                fig.tight_layout(h_pad=2)
            if len(feat_name) == 2:
                feat1_old = feat1.copy()
                feat2_old = feat2.copy()
                feat1 = []
                feat2 = []
                [feat1.append(x) for x in feat1_old if x not in feat1]
                [feat2.append(x) for x in feat2_old if x not in feat2]
                test_matrix = np.zeros([len(feat1), len(feat2)])
                for j in range(len(feat1)):
                    for h in range(len(feat2)):
                        test_index = 0
                        for m in range(len(test)):
                            if feat1_old[m] == feat1[j] and feat2_old[m] == feat2[h]:
                                test_index = m
                                break
                        test_matrix[j, h] = test[test_index]
                fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
                plt.pcolormesh(test_matrix, cmap=plt.cm.PuBuGn)
                plt.colorbar()
                ax.set_xlabel('Parameter sweep ' + feat_name[1], fontsize=14)
                ax.set_ylabel('Parameter sweep ' + feat_name[0], fontsize=14)
                ax.set_title('Test score assessment per parameter sweep with ' + algorithm[i].upper() + ' algorithm',
                             fontsize=24)
                ax.set_xticks(np.arange(0.5, len(feat2) + 0.5), labels=feat2, fontsize=14)
                ax.set_yticks(np.arange(0.5, len(feat1) + 0.5), labels=feat1, fontsize=14)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                for j in range(len(feat1)):
                    for h in range(len(feat2)):
                        ax.text(h + 0.5, j + 0.5, str(round(test_matrix[j, h], 4)),
                                ha="center", va="center", color="k", fontweight='bold', fontsize=10)
                fig.tight_layout(h_pad=2)
            if len(feat_name) == 3:
                feat1_old = feat1.copy()
                feat2_old = feat2.copy()
                feat3_old = feat3.copy()
                feat1 = []
                feat2 = []
                feat3 = []
                [feat1.append(x) for x in feat1_old if x not in feat1]
                [feat2.append(x) for x in feat2_old if x not in feat2]
                [feat3.append(x) for x in feat3_old if x not in feat3]
                if len(feat1) <= len(feat2) and len(feat1) <= len(feat3):
                    featmin = feat1
                    featx = feat2
                    featy = feat3
                    featmin_old = feat1_old
                    featx_old = feat2_old
                    featy_old = feat3_old
                    feat_name = [feat_name[0], feat_name[1], feat_name[2]]
                elif len(feat2) <= len(feat1) and len(feat2) <= len(feat3):
                    featmin = feat2
                    featx = feat1
                    featy = feat3
                    featmin_old = feat2_old
                    featx_old = feat1_old
                    featy_old = feat3_old
                    feat_name = [feat_name[1], feat_name[0], feat_name[2]]
                else:
                    featmin = feat3
                    featx = feat1
                    featy = feat2
                    featmin_old = feat3_old
                    featx_old = feat1_old
                    featy_old = feat2_old
                    feat_name = [feat_name[2], feat_name[0], feat_name[1]]
                fig, axes = plt.subplots(math.ceil(len(featmin) / self.nfeat), self.nfeat,
                                         figsize=(self.fig_width, self.fig_height))
                spare_axes = self.nfeat - len(featmin) % self.nfeat
                if spare_axes == self.nfeat:
                    spare_axes = 0
                for axis in range(self.nfeat - 1, self.nfeat - 1 - spare_axes, -1):
                    if (math.ceil(len(featmin) / self.nfeat) - 1) == 0:
                        fig.delaxes(axes[axis])
                    else:
                        fig.delaxes(axes[math.ceil(len(featmin) / self.nfeat) - 1, axis])
                ax = axes.ravel()
                for p in range(len(featmin)):
                    test_matrix = np.zeros([len(featy), len(featx)])
                    for j in range(len(featy)):
                        for h in range(len(featx)):
                            test_index = 0
                            for m in range(len(test)):
                                if featy_old[m] == featy[j] and featx_old[m] == featx[h] and featmin_old[m] == featmin[p]:
                                    test_index = m
                                    break
                            test_matrix[j, h] = test[test_index]
                    pcm = ax[p].pcolormesh(test_matrix, cmap=plt.cm.PuBuGn)
                    divider = make_axes_locatable(ax[p])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(pcm, cax=cax, orientation='vertical')
                    ax[p].set_xlabel('Parameter sweep ' + feat_name[1], fontsize=14)
                    ax[p].set_ylabel('Parameter sweep ' + feat_name[2], fontsize=14)
                    ax[p].set_title('Parameter sweep with fixed ' + feat_name[0] + ' = ' + str(featmin[p]), fontsize=16)
                    ax[p].set_xticks(np.arange(0.5, len(featx) + 0.5), labels=featx, fontsize=12)
                    ax[p].set_yticks(np.arange(0.5, len(featy) + 0.5), labels=featy, fontsize=12)
                    plt.setp(ax[p].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    for j in range(len(featy)):
                        for h in range(len(featx)):
                            ax[p].text(h + 0.5, j + 0.5, str(round(test_matrix[j, h], 4)),
                                       ha="center", va="center", color="k", fontweight='bold', fontsize=10)
                fig.suptitle('Test score assessment per parameter sweep with ' + algorithm[i].upper() + ' algorithm',
                             fontsize=24)
                plt.subplots_adjust(top=0.85)
                fig.tight_layout(h_pad=2)
            plt.savefig('Parameter sweep ' + algorithm[i].upper() + ' algorithm.png', bbox_inches='tight')
            plt.clf()
