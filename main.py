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


# INPUT PARAMETERS
fig_width = 20
fig_height = 10
bar_width = 0.25
percentile = 0.02
# ALGORITHM ENABLE FLAGS
knn = 0
logreg = 0
linearsvc = 0
naivebayes = 0
tree = 0
forest = 0
gradient = 0
svm = 1
mlp = 0

# READ INPUT DATA
sourcedf = pd.read_csv('cardio_train.csv')
print("Full source data from CSV type: {} and shape: {}".format(type(sourcedf), sourcedf.shape))
cardio0 = sourcedf.loc[sourcedf['cardio'] == 0, :]
cardio1 = sourcedf.loc[sourcedf['cardio'] == 1, :]
print("Source cases cardio = 0 type: {} and shape: {}".format(type(cardio0), cardio0.shape))
print("Source cases cardio = 1 type: {} and shape: {} \n".format(type(cardio1), cardio1.shape))

# HISTOGRAM FOR ORIGINAL CSV DATA
fig, axes = plt.subplots(math.ceil(sourcedf.shape[1] / 2), 2, figsize=(fig_width, fig_height))
if sourcedf.shape[1] % 2 == 1:
    fig.delaxes(axes[math.ceil(sourcedf.shape[1] / 2) - 1, 1])
ax = axes.ravel()
for i in range(sourcedf.shape[1]):
    ax[i].hist(cardio0.iloc[:, i], histtype='stepfilled', bins=25, alpha=0.25, color="#0000FF", lw=0)
    ax[i].hist(cardio1.iloc[:, i], histtype='stepfilled', bins=25, alpha=0.25, color='#FF0000', lw=0)
    ax[i].set_title(sourcedf.keys()[i], fontsize=10, fontweight='bold')
    ax[i].grid(visible=True)
    ax[i].tick_params(axis='both', labelsize=8)
    ax[i].set_ylabel("Frequency", fontsize=8)
    ax[i].set_xlabel("Feature magnitude", fontsize=8)
ax[0].legend(["cardio0", "cardio1"], loc="best")
plt.savefig('Original histogram.png', bbox_inches='tight')
plt.clf()

# MODIFY INPUT DATA TO ELIMINATE OUTLIERS, DUPLICATES AND WRONG DATASETS
df = sourcedf.drop('id', axis=1)
print("Scrubber data after eliminating non-meaningful columns type: {} and shape: {}".format(type(df), df.shape))
df.drop_duplicates(keep='first', inplace=True)
print("Scrubber data after eliminating duplicates type: {} and shape: {}".format(type(df), df.shape))
df_qmin = df.quantile(percentile)
df_qmax = df.quantile(1 - percentile)
for i in range(len(df.keys())):
    if min(df.iloc[:, i]) == 0 and max(df.iloc[:, i] <= 3):
        continue
    else:
        df = df.loc[df[df.keys()[i]] >= df_qmin[i], :]
        df = df.loc[df[df.keys()[i]] <= df_qmax[i], :]
print("Scrubber data after eliminating outliers type: {} and shape: {}".format(type(df), df.shape))
df.replace('', np.nan, inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df.reset_index(drop=True, inplace=True)
print("Scrubber data after eliminating empty datasets type: {} and shape: {}".format(type(df), df.shape))
index_to_drop = []
for i in range(df.shape[0]):
    if df.iloc[i, df.columns.get_loc('ap_lo')] > df.iloc[i, df.columns.get_loc('ap_hi')]:
        index_to_drop.append(i)
df.drop(index_to_drop, inplace=True)
df.reset_index(drop=True, inplace=True)
print("Scrubber data after eliminating non-consistent datasets type: {} and shape: {}".format(type(df), df.shape))

# HISTOGRAM FOR MODIFIED DATA
fig, axes = plt.subplots(math.ceil(df.shape[1] / 2), 2, figsize=(fig_width, fig_height))
if df.shape[1] % 2 == 1:
    fig.delaxes(axes[math.ceil(df.shape[1] / 2) - 1, 1])
ax = axes.ravel()
cardio0 = df.loc[df['cardio'] == 0, :]
cardio1 = df.loc[df['cardio'] == 1, :]
print("Scrubber data cardio = 0 type: {} and shape: {}".format(type(cardio0), cardio0.shape))
print("Scrubber data  cardio = 1 type: {} and shape: {} \n".format(type(cardio1), cardio1.shape))
for i in range(df.shape[1]):
    ax[i].hist(cardio0.iloc[:, i], histtype='stepfilled', bins=25, alpha=0.25, color="#0000FF", lw=0)
    ax[i].hist(cardio1.iloc[:, i], histtype='stepfilled', bins=25, alpha=0.25, color='#FF0000', lw=0)
    ax[i].set_title(df.keys()[i], fontsize=10, fontweight='bold')
    ax[i].grid(visible=True)
    ax[i].tick_params(axis='both', labelsize=8)
    ax[i].set_ylabel("Frequency", fontsize=8)
    ax[i].set_xlabel("Feature magnitude", fontsize=8)
ax[0].legend(["cardio0", "cardio1"], loc="best")
plt.savefig('Scrubber histogram.png', bbox_inches='tight')
plt.clf()

# SPLIT BETWEEN TRAIN AND TEST DATA
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, [-1]], test_size=0.2, shuffle=True,
                                                    stratify=df.iloc[:, [-1]], random_state=1)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
print("X_train type: {} and shape: {}".format(type(X_train), X_train.shape))
print("X_test type: {} and shape: {}".format(type(X_test), X_test.shape))
print("y_train type: {} and shape: {}".format(type(y_train), y_train.shape))
print("y_test type: {} and shape: {} \n".format(type(y_test), y_test.shape))

# CHECK THE OUTPUT CLASS DISTRIBUTION IN BOTH TRAIN AND TEST DATA
y_train_cardio1 = y_train[y_train == 1]
y_train_cardio0 = y_train[y_train == 0]
y_test_cardio1 = y_test[y_test == 1]
y_test_cardio0 = y_test[y_test == 0]
print("y_train_cardio1 type: {} and shape: {}".format(type(y_train_cardio1), y_train_cardio1.shape))
print("y_test_cardio1 type: {} and shape: {}".format(type(y_test_cardio1), y_test_cardio1.shape))
print("y_train_cardio0 type: {} and shape: {}".format(type(y_train_cardio0), y_train_cardio0.shape))
print("y_test_cardio0 type: {} and shape: {} \n".format(type(y_test_cardio0), y_test_cardio0.shape))

# PLOT THE OUTPUT CLASS DISTRIBUTION IN BOTH TRAIN AND TEST DATA
plt.subplots(figsize=(fig_width, fig_height))
plt.bar([1, 2], [y_train_cardio0.shape[0], y_test_cardio0.shape[0]],
        color='r', width=bar_width, edgecolor='black', label='cardio=0')
plt.bar([1 + bar_width, 2 + bar_width], [y_train_cardio1.shape[0], y_test_cardio1.shape[0]],
        color='b', width=bar_width, edgecolor='black', label='cardio=1')
plt.xticks([1 + bar_width / 2, 2 + bar_width / 2], ['cardio train data', 'cardio test data'], ha='center')
plt.text(1 - bar_width / 4, y_train_cardio0.shape[0] + 100, str(y_train_cardio0.shape[0]), fontsize=20)
plt.text(1 + 3 * bar_width / 4, y_train_cardio1.shape[0] + 100, str(y_train_cardio1.shape[0]), fontsize=20)
plt.text(2 - bar_width / 4, y_test_cardio0.shape[0] + 100, str(y_test_cardio0.shape[0]), fontsize=20)
plt.text(2 + 3 * bar_width / 4, y_test_cardio1.shape[0] + 100, str(y_test_cardio1.shape[0]), fontsize=20)
plt.title('Cardiovascular disease counting cases', fontsize=24)
plt.xlabel('Concepts', fontweight='bold', fontsize=14)
plt.ylabel('Count cases', fontweight='bold', fontsize=14)
plt.legend()
plt.grid()
plt.savefig('Count_cases.png', bbox_inches='tight')
plt.clf()

# STANDARDIZE TRAIN AND TEST DATA
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("X_train_scaled type: {} and shape: {}".format(type(X_train_scaled), X_train_scaled.shape))
print("X_test_scaled type: {} and shape: {} \n".format(type(X_test_scaled), X_test_scaled.shape))

# APPLY PCA ALGORITHM IN TRAIN DATA
pca = PCA(n_components=X_train_scaled.shape[1])
pca.fit(X_train_scaled)
X_train_scaled_pca = pca.transform(X_train_scaled)
X_test_scaled_pca = pca.transform(X_test_scaled)
print("X_train_scaled PCA type: {} and shape: {}".format(type(X_train_scaled_pca), X_train_scaled_pca.shape))
print("X_test_scaled PCA type: {} and shape: {}".format(type(X_test_scaled_pca), X_test_scaled_pca.shape))
X_train_scaled_pca_cardio1 = np.zeros([y_train_cardio1.shape[0], X_train_scaled_pca.shape[1]])
X_train_scaled_pca_cardio0 = np.zeros([y_train_cardio0.shape[0], X_train_scaled_pca.shape[1]])
h1 = 0
h0 = 0
for i in range(len(y_train)):
    if y_train[i] == 1:
        X_train_scaled_pca_cardio1[h1, :] = X_train_scaled_pca[i, :]
        h1 += 1
    elif y_train[i] == 0:
        X_train_scaled_pca_cardio0[h0, :] = X_train_scaled_pca[i, :]
        h0 += 1
print("X_train_scaled PCA cardio1 type: {} and shape: {}"
      .format(type(X_train_scaled_pca_cardio1), X_train_scaled_pca_cardio1.shape))
print("X_train_scaled PCA cardio0 type: {} and shape: {}"
      .format(type(X_train_scaled_pca_cardio0), X_train_scaled_pca_cardio0.shape))
print("PCA component shape: {} \n".format(pca.components_.shape))

# PLOT FIRST AND SECOND PCA BREAKDOWN FROM ALL INPUT FEATURES
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
plt.pcolormesh(pca.components_, cmap=plt.cm.cool)
plt.colorbar()
pca_range = [x + 0.5 for x in range(pca.components_.shape[1])]
plt.xticks(pca_range, X_train.keys(), rotation=60, ha='center')
ax.xaxis.tick_top()
str_pca = []
for i in range(X_train.shape[1]):
    str_pca.append('Component ' + str(i + 1))
plt.yticks(pca_range, str_pca)
plt.xlabel("Feature", weight='bold', fontsize=14)
plt.ylabel("Principal components", weight='bold', fontsize=14)
plt.savefig('PCA_scaled_breakdown.png', bbox_inches='tight')
plt.clf()

# PLOT FIRST AND SECOND PCA INTERCONNECTION
plt.subplots(figsize=(fig_width, fig_height))
plt.scatter(X_train_scaled_pca_cardio1[:, 0], X_train_scaled_pca_cardio1[:, 1],
            s=10, marker='^', c='red', label='cardio=1')
plt.scatter(X_train_scaled_pca_cardio0[:, 0], X_train_scaled_pca_cardio0[:, 1],
            s=10, marker='o', c='blue', label='cardio=0')
plt.title('Cardiovascular disease modelling', fontsize=20, fontweight='bold')
plt.xlabel('First PCA', fontsize=14)
plt.ylabel('Second PCA', fontsize=14)
plt.legend()
plt.grid()
plt.savefig('PCA_scaled_First_Second.png', bbox_inches='tight')
plt.clf()

# PLOT PCA SCREE PLOT
fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
ax2 = ax1.twinx()
label1 = ax1.plot(range(1, len(pca.components_) + 1), pca.explained_variance_ratio_,
                  'ro-', linewidth=2, label='Individual PCA variance')
label2 = ax2.plot(range(1, len(pca.components_) + 1), np.cumsum(pca.explained_variance_ratio_),
                  'b^-', linewidth=2, label='Cumulative PCA variance')
plt.title('Scree Plot', fontsize=20, fontweight='bold')
ax1.set_xlabel('Principal Components', fontsize=14)
ax1.set_ylabel('Proportion of Variance Explained', fontsize=14, color='r')
ax2.set_ylabel('Cumulative Proportion of Variance Explained', fontsize=14, color='b')
lA = label1+label2
lB = [lA[0].get_label(), lA[1].get_label()]
ax1.legend(lA, lB, loc='upper center')
ax1.grid(visible=True)
ax2.grid(visible=True)
plt.savefig('PCA_scaled_ScreePlot.png', bbox_inches='tight')
plt.clf()

# KNN
if knn == 1:
    ini_time = time.time()
    model = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    model.fit(X_train, y_train)
    print('SCORE WITH KNN (K = 1) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

    ini_time = time.time()
    model = KNeighborsClassifier(n_neighbors=7, weights='uniform')
    model.fit(X_train, y_train)
    print('SCORE WITH KNN (K = 7) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

    ini_time = time.time()
    model = KNeighborsClassifier(n_neighbors=25, weights='uniform')
    model.fit(X_train, y_train)
    print('SCORE WITH KNN (K = 25) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

# LOGISTIC REGRESSION
if logreg == 1:
    ini_time = time.time()
    model = LogisticRegression(random_state=0, C=0.01, max_iter=10000)
    model.fit(X_train, y_train)
    print('SCORE WITH LOGISTIC REGRESSION C=0.01 ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

    ini_time = time.time()
    model = LogisticRegression(random_state=0, C=1000, max_iter=10000)
    model.fit(X_train, y_train)
    print('SCORE WITH LOGISTIC REGRESSION C=1000 ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

# LINEAR SVC
if linearsvc == 1:
    ini_time = time.time()
    model = LinearSVC(random_state=0, C=0.01, max_iter=1000)
    model.fit(X_train, y_train)
    print('SCORE WITH LINEAR SVC C=0.01 ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

    ini_time = time.time()
    model = LinearSVC(random_state=0, C=1000, max_iter=20000)
    model.fit(X_train, y_train)
    print('SCORE WITH LINEAR SVC C=1000 ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

# NAIVE BAYES
if naivebayes == 1:
    ini_time = time.time()
    model = GaussianNB()
    model.fit(X_train, y_train)
    print('SCORE WITH NAIVE BAYES ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

# DECISION TREE
if tree == 1:
    ini_time = time.time()
    model = DecisionTreeClassifier(random_state=0, max_depth=5)
    model.fit(X_train, y_train)
    print('SCORE WITH DECISION TREE (MAX DEPTH = 5) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

    model = DecisionTreeClassifier(random_state=0, max_depth=25)
    ini_time = time.time()
    model.fit(X_train, y_train)
    print('SCORE WITH DECISION TREE (MAX DEPTH = 25) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

# RANDOM FOREST
if forest == 1:
    ini_time = time.time()
    model = RandomForestClassifier(random_state=0, n_estimators=5, max_features=11, max_depth=25)
    model.fit(X_train, y_train)
    print('SCORE WITH RANDOM FOREST (ESTIMATORS 5, MAX FEATURES 11 AND MAX DEPTH 25) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

    ini_time = time.time()
    model = RandomForestClassifier(random_state=0, n_estimators=5, max_features=4, max_depth=25)
    model.fit(X_train, y_train)
    print('SCORE WITH RANDOM FOREST (ESTIMATORS 5, MAX FEATURES 4 AND MAX DEPTH 25) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

    ini_time = time.time()
    model = RandomForestClassifier(random_state=0, n_estimators=100, max_features=4, max_depth=25)
    model.fit(X_train, y_train)
    print('SCORE WITH RANDOM FOREST (ESTIMATORS 100, MAX FEATURES 4 AND MAX DEPTH 25) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

    ini_time = time.time()
    model = RandomForestClassifier(random_state=0, n_estimators=100, max_features=4, max_depth=8)
    model.fit(X_train, y_train)
    print('SCORE WITH RANDOM FOREST (ESTIMATORS 100, MAX FEATURES 4 AND MAX DEPTH 8) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

    ini_time = time.time()
    model = RandomForestClassifier(random_state=0, n_estimators=200, max_features=4, max_depth=8)
    model.fit(X_train, y_train)
    print('SCORE WITH RANDOM FOREST (ESTIMATORS 200, MAX FEATURES 4 AND MAX DEPTH 8) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

# GRADIENT BOOSTING
if gradient == 1:
    ini_time = time.time()
    model = GradientBoostingClassifier(random_state=0, n_estimators=500, learning_rate=1, max_depth=4)
    model.fit(X_train, y_train)
    print('SCORE WITH GRADIENT BOOSTING (ESTIMATORS 500, MAX DEPTH 4 AND LEARNING RATE 1) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

    ini_time = time.time()
    model = GradientBoostingClassifier(random_state=0, n_estimators=25, learning_rate=1, max_depth=4)
    model.fit(X_train, y_train)
    print('SCORE WITH GRADIENT BOOSTING (ESTIMATORS 25, MAX DEPTH 4 AND LEARNING RATE 1) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

    ini_time = time.time()
    model = GradientBoostingClassifier(random_state=0, n_estimators=25, learning_rate=0.2, max_depth=4)
    model.fit(X_train, y_train)
    print('SCORE WITH GRADIENT BOOSTING (ESTIMATORS 25, MAX DEPTH 4 AND LEARNING RATE 0.2) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

# NEURAL NETWORK
if mlp == 1:
    ini_time = time.time()
    model = MLPClassifier(random_state=0, activation='tanh', hidden_layer_sizes=[50, 50], alpha=0.01, max_iter=1000)
    model.fit(X_train, y_train)
    print('SCORE WITH NEURAL NETWORK 2 HIDDEN LAYERS OF 50 NODES EACH AND ALPHA 0.01 ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

    ini_time = time.time()
    model = MLPClassifier(random_state=0, activation='tanh', hidden_layer_sizes=100, alpha=0.01)
    model.fit(X_train, y_train)
    print('SCORE WITH NEURAL NETWORK 1 HIDDEN LAYERS OF 100 NODES AND ALPHA 0.01 ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

    ini_time = time.time()
    model = MLPClassifier(random_state=0, activation='tanh', hidden_layer_sizes=100, alpha=10)
    model.fit(X_train, y_train)
    print('SCORE WITH NEURAL NETWORK 1 HIDDEN LAYERS OF 100 NODES AND ALPHA 10 ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))

# SVM
if svm == 1:
    ini_time = time.time()
    model = SVC(random_state=0, kernel='rbf', C=0.1, gamma=1)
    model.fit(X_train, y_train)
    print('SCORE WITH SVM (KERNEL RBF, C = 0.1 AND GAMMA = 1) ALGORITHM')
    print('TRAIN dataset: {}'.format(str(model.score(X_train, y_train))))
    print('TEST dataset: {}'.format(str(model.score(X_test, y_test))))
    model.fit(X_train_scaled, y_train)
    print('Scaling TRAIN dataset: {}'.format(str(model.score(X_train_scaled, y_train))))
    print('Scaling TEST dataset: {}'.format(str(model.score(X_test_scaled, y_test))))
    print('Execution time [seconds]: {}\n'.format(str(time.time() - ini_time)))