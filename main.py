from supervised_algorithms import SupervisedAlgorithms
from data_preprocessing import DataPreprocessing
from data_visualization import DataPlot
from pca_analysis import PCAanalysis
import utils
import pandas as pd
import numpy as np


cross_validation = 0
sourcedf = pd.read_csv('cardio_train.csv')
print("Full source data from CSV type: {} and shape: {}".format(type(sourcedf), sourcedf.shape))
supervised = SupervisedAlgorithms()
visualization = DataPlot()
preprocessing = DataPreprocessing(sourcedf.copy())

df_unscaled = preprocessing.data_scrubbing(columns_to_remove='id', concept1='ap_lo', concept2='ap_hi',
                                           encodings=['gender', 'cholesterol', 'gluc'], class_column_name='cardio',
                                           max_filter=True, min_filter=True, max_threshold=1, min_threshold=0)
list_features = df_unscaled.keys()[:-1]
visualization.boxplot(dataset=sourcedf.iloc[:, 1:], plot_name='Original boxplot', max_features_row=3)
visualization.binary_class_histogram(dataset=sourcedf, class_column_name='cardio', plot_name='Original histogram',
                                     ncolumns=3)

visualization.boxplot(dataset=df_unscaled, plot_name='Scrubbed boxplot', max_features_row=6)
visualization.binary_class_histogram(dataset=df_unscaled, class_column_name='cardio', plot_name='Scrubbed histogram',
                                     ncolumns=3)

df_train_scaled, target_train, df_test_scaled, target_test = supervised.train_test_split(
    feature_data=df_unscaled.iloc[:, :-1], class_data=df_unscaled.iloc[:, [-1]], test_size=0.2, algorithm='std')

visualization.plot_output_class_distribution(target_train, target_test)
pca = PCAanalysis(list_features=list_features, dataset=df_train_scaled, ncomps=len(list_features), target=target_train)
pca.apply_pca()


if cross_validation:
    algorithm = ['tree', 'forest', 'gradient', 'mlp']
    scale = ['', '', '', 'std']
    params = [
        {'classifier': [], 'preprocessing': [], 'classifier__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {'classifier': [], 'preprocessing': [], 'classifier__n_estimators': [25, 50],
         'classifier__max_depth': [6, 7], 'classifier__max_features': [4, 8]},
        {'classifier': [], 'preprocessing': [], 'classifier__n_estimators': [10, 15, 25, 35, 50],
         'classifier__max_depth': [1, 2, 3, 4, 5], 'classifier__learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]},
        {'classifier': [], 'preprocessing': [], 'classifier__activation': ['tanh', 'relu', 'logistic'],
         'classifier__alpha': [0.005, 0.01, 0.1, 0.25, 0.5],
         'classifier__hidden_layer_sizes': [25, 50, 100, [5, 5], [25, 25], [50, 50]]}]
    grid = supervised.cross_grid_validation(algorithm, scale, param_grid=params, nfolds=5)
    pd_grid = pd.DataFrame(grid.cv_results_)
    visualization.param_sweep_plot(algorithm, params=pd_grid['params'], test_score=pd_grid['mean_test_score'])
else:
    alg = []
    pars = []
    results = np.zeros([0, 0])  # Initialize numpy array to fill in dynamically (first writing requires update shape)
    alg, pars, results = supervised.apply_algorithm('knn', {'n_neighbors': 1}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('knn', {'n_neighbors': 7}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('knn', {'n_neighbors': 25}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('linearsvc', {'C': 0.01}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('linearsvc', {'C': 1000}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('logreg', {'C': 0.01}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('logreg', {'C': 1000}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('naivebayes', {}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('tree', {'max_depth': 25}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('tree', {'max_depth': 5}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('forest', {'n_estimators': 5, 'max_features': 11,
                                                               'max_depth': 25}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('forest', {'n_estimators': 5, 'max_features': 4,
                                                               'max_depth': 25}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('forest', {'n_estimators': 100, 'max_features': 4,
                                                               'max_depth': 25}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('forest', {'n_estimators': 100, 'max_features': 4,
                                                               'max_depth': 8}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('forest', {'n_estimators': 200, 'max_features': 4,
                                                               'max_depth': 8}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('gradient', {'n_estimators': 500, 'learning_rate': 1,
                                                                 'max_depth': 4}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('gradient', {'n_estimators': 25, 'learning_rate': 1,
                                                                 'max_depth': 4}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('gradient', {'n_estimators': 25, 'learning_rate': 0.2,
                                                                 'max_depth': 4}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('mlp', {'activation': 'tanh', 'alpha': 0.01,
                                                            'hidden_layer_sizes': [50, 50]}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('mlp', {'activation': 'tanh', 'alpha': 0.01,
                                                            'hidden_layer_sizes': 100}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('mlp', {'activation': 'tanh', 'alpha': 10,
                                                            'hidden_layer_sizes': 100}, alg, pars, results)
    alg, pars, results = supervised.apply_algorithm('svm', {'kernel': 'rbf', 'C': 0.1, 'gamma': 1}, alg, pars, results)

    utils.write_results_excel_file('results.xlsx', alg, pars, results)
