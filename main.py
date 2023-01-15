from supervised_algorithms import SupervisedAlgorithms
from data_preprocessing import DataPreprocessing
from data_visualization import BinaryClassDataPlot
from pca_analysis import PCAanalysis
import pandas as pd


sourcedf = pd.read_csv('cardio_train.csv')
print("Full source data from CSV type: {} and shape: {}".format(type(sourcedf), sourcedf.shape))
supervised = SupervisedAlgorithms()
visualization = BinaryClassDataPlot()
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

dataset, target = supervised.train_test_split(feature_data=df_unscaled.iloc[:, :-1],
                                              class_data=df_unscaled.iloc[:, [-1]], test_size=0.2, algorithm='standard')

pca = PCAanalysis(list_features, dataset, target)
pca.apply_pca(ncomps=len(list_features))

supervised.apply_algorithm(algorithm='knn', params={'n_neighbors': 1})
supervised.apply_algorithm(algorithm='knn', params={'n_neighbors': 7})
supervised.apply_algorithm(algorithm='knn', params={'n_neighbors': 25})
supervised.apply_algorithm(algorithm='linearsvc', params={'C': 0.01})
supervised.apply_algorithm(algorithm='linearsvc', params={'C': 1000})
supervised.apply_algorithm(algorithm='logreg', params={'C': 0.01})
supervised.apply_algorithm(algorithm='logreg', params={'C': 1000})
supervised.apply_algorithm(algorithm='naivebayes', params={})
supervised.apply_algorithm(algorithm='tree', params={'max_depth': 25})
supervised.apply_algorithm(algorithm='tree', params={'max_depth': 5})
supervised.apply_algorithm(algorithm='forest', params={'n_estimators': 5, 'max_features': 11, 'max_depth': 25})
supervised.apply_algorithm(algorithm='forest', params={'n_estimators': 5, 'max_features': 4, 'max_depth': 25})
supervised.apply_algorithm(algorithm='forest', params={'n_estimators': 100, 'max_features': 4, 'max_depth': 25})
supervised.apply_algorithm(algorithm='forest', params={'n_estimators': 100, 'max_features': 4, 'max_depth': 8})
supervised.apply_algorithm(algorithm='forest', params={'n_estimators': 200, 'max_features': 4, 'max_depth': 8})
supervised.apply_algorithm(algorithm='gradient', params={'n_estimators': 500, 'learning_rate': 1, 'max_depth': 4})
supervised.apply_algorithm(algorithm='gradient', params={'n_estimators': 25, 'learning_rate': 1, 'max_depth': 4})
supervised.apply_algorithm(algorithm='gradient', params={'n_estimators': 25, 'learning_rate': 0.2, 'max_depth': 4})
supervised.apply_algorithm(algorithm='mlp', params={'activation': 'tanh', 'alpha': 0.01,
                                                    'hidden_layer_sizes': [50, 50]})
supervised.apply_algorithm(algorithm='mlp', params={'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': 100})
supervised.apply_algorithm(algorithm='mlp', params={'activation': 'tanh', 'alpha': 10, 'hidden_layer_sizes': 100})
supervised.apply_algorithm(algorithm='svm', params={'kernel': 'rbf', 'C': 0.1, 'gamma': 1})
supervised.write_results_excel_file('results.xlsx')
