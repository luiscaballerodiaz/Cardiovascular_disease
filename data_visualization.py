import matplotlib.pyplot as plt
import math
import numpy as np


class BinaryClassDataPlot:
    """Class to visualize the input data split per binary output class"""

    def __init__(self):
        self.fig_width = 20
        self.fig_height = 10
        self.bar_width = 0.25

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
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.clf()
