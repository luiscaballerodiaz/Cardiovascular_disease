import matplotlib.pyplot as plt
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
