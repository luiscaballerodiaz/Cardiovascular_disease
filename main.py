import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

k = 15
df = pd.read_csv('cardio_train.csv')
df = df.sample(frac=1, random_state=1)
header = df.keys()
traindf = df.iloc[:int(0.8 * len(df)), :]
traindf = traindf.loc[(traindf['ap_lo'] < 300) & (traindf['ap_lo'] > 0)]
traindf = traindf.loc[(traindf['ap_hi'] < 300) & (traindf['ap_hi'] > 50)]
traindf = traindf.loc[traindf['age'] > 14000]
testdf = df.iloc[1 + int(0.2 * len(df)):, :]

for i in range(1, len(header) - 1):
    traindf_cardio1 = traindf.loc[traindf['cardio'] == 1, header[i]]
    traindf_cardio0 = traindf.loc[traindf['cardio'] == 0, header[i]]
    plt.scatter(range(len(traindf_cardio1)), traindf_cardio1, s=1, c='red', label='cardio=1')
    plt.scatter(range(len(traindf_cardio0)), traindf_cardio0, s=1, c='blue', label='cardio=0')
    plt.scatter(len(traindf_cardio1) / 2, traindf_cardio1.mean(), s=50, c='green', label='mean cardio=1')
    plt.scatter(len(traindf_cardio1) / 2, traindf_cardio0.mean(), s=50, c='yellow', label='mean cardio=0')
    plt.title('Cardiovascular disease modelling', fontsize=24)
    plt.xlabel('Patients', fontsize=14)
    plt.ylabel(header[i], fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(header[i] + '.png', bbox_inches='tight')
    plt.clf()
x=3
trainmean = []
trainstd = []
trainmin = []
trainmax = []
traindf_zscore = traindf[:]
testdf_zscore = testdf[:]
traindf_norm = traindf[:]
testdf_norm = testdf[:]
for h in range(len(traindf.keys())):
    trainmean.append(traindf[header[h]].mean())
    trainstd.append(traindf[header[h]].std())
    trainmin.append(traindf[header[h]].min())
    trainmax.append(traindf[header[h]].max())
    traindf_zscore[header[h]] = (traindf[header[h]] - trainmean[h]) / trainstd[h]
    testdf_zscore[header[h]] = (testdf[header[h]] - trainmean[h]) / trainstd[h]
    traindf_norm[header[h]] = (traindf[header[h]] - trainmin[h]) / (trainmax[h] - trainmin[h])
    testdf_norm[header[h]] = (testdf[header[h]] - trainmin[h]) / (trainmax[h] - trainmin[h])

traindf_zscore_cov = traindf_zscore.cov()
traindf_zscore_cov = traindf_zscore_cov.loc[traindf_zscore_cov['cardio'] > 0.15, 'cardio']
traindf_norm_mod = traindf_norm.loc[:, traindf_zscore_cov.keys()]
traindf_zscore_mod = traindf_zscore.loc[:, traindf_zscore_cov.keys()]
testdf_norm_mod = testdf_norm.loc[:, traindf_zscore_cov.keys()]
testdf_zscore_mod = testdf_zscore.loc[:, traindf_zscore_cov.keys()]
traindf_zscore_cov = traindf_zscore_cov.iloc[:-1]
traindf_zscore_cov_pu = traindf_zscore_cov / sum(traindf_zscore_cov)
traindf_zscore_mod['total'] = 0
testdf_zscore_mod['total'] = 0
traindf_norm_mod['total'] = 0
testdf_norm_mod['total'] = 0
for h in range(len(traindf_zscore_cov_pu)):
    traindf_zscore_mod['total'] += traindf_zscore_cov_pu[h] * traindf_zscore_mod[traindf_zscore_cov_pu.keys()[h]]
    testdf_zscore_mod['total'] += traindf_zscore_cov_pu[h] * testdf_zscore_mod[traindf_zscore_cov_pu.keys()[h]]
    traindf_norm_mod['total'] += traindf_zscore_cov_pu[h] * traindf_norm_mod[traindf_zscore_cov_pu.keys()[h]]
    testdf_norm_mod['total'] += traindf_zscore_cov_pu[h] * testdf_norm_mod[traindf_zscore_cov_pu.keys()[h]]
    traindf_norm_mod[traindf_zscore_cov_pu.keys()[h]] *= (1 + traindf_zscore_cov_pu[h])
    testdf_norm_mod[traindf_zscore_cov_pu.keys()[h]] *= (1 + traindf_zscore_cov_pu[h])
    traindf_zscore_mod[traindf_zscore_cov_pu.keys()[h]] *= (1 + traindf_zscore_cov_pu[h])
    testdf_zscore_mod[traindf_zscore_cov_pu.keys()[h]] *= (1 + traindf_zscore_cov_pu[h])

traindf_cardio1 = traindf_zscore_mod.loc[traindf_zscore_mod['cardio'] > 0, 'total']
traindf_cardio0 = traindf_zscore_mod.loc[traindf_zscore_mod['cardio'] < 0, 'total']
plt.scatter(range(len(traindf_cardio1)), traindf_cardio1, s=1, c='red', label='cardio=1')
plt.scatter(range(len(traindf_cardio0)), traindf_cardio0, s=1, c='blue', label='cardio=0')
plt.scatter(len(traindf_cardio1) / 2, traindf_cardio1.mean(), s=50, c='green', label='mean cardio=1')
plt.scatter(len(traindf_cardio1) / 2, traindf_cardio0.mean(), s=50, c='yellow', label='mean cardio=0')
plt.title('Cardiovascular disease modelling', fontsize=24)
plt.xlabel('Patients', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.legend()
plt.grid()
plt.savefig('NewFeatureStandardized.png', bbox_inches='tight')
plt.clf()

traindf_cardio1 = traindf_norm_mod.loc[traindf_norm_mod['cardio'] == 1, 'total']
traindf_cardio0 = traindf_norm_mod.loc[traindf_norm_mod['cardio'] == 0, 'total']
plt.scatter(range(len(traindf_cardio1)), traindf_cardio1, s=1, c='red', label='cardio=1')
plt.scatter(range(len(traindf_cardio0)), traindf_cardio0, s=1, c='blue', label='cardio=0')
plt.scatter(len(traindf_cardio1) / 2, traindf_cardio1.mean(), s=50, c='green', label='mean cardio=1')
plt.scatter(len(traindf_cardio1) / 2, traindf_cardio0.mean(), s=50, c='yellow', label='mean cardio=0')
plt.title('Cardiovascular disease modelling', fontsize=24)
plt.xlabel('Patients', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.legend()
plt.grid()
plt.savefig('NewFeatureNormalized.png', bbox_inches='tight')
plt.clf()

y_train = traindf.iloc[:, -1].to_numpy()
y_test = testdf.iloc[:, -1].to_numpy()

X_train = traindf.iloc[:, 1:-1].to_numpy()
X_test = testdf.iloc[:, 1:-1].to_numpy()
knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
knn.fit(X_train, y_train)
print('Score with unmodified data: ' + str(knn.score(X_test, y_test)))

X_train = traindf_zscore.iloc[:, 1:-1].to_numpy()
X_test = testdf_zscore.iloc[:, 1:-1].to_numpy()
knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
knn.fit(X_train, y_train)
print('Score with standardized data: ' + str(knn.score(X_test, y_test)))

X_train = traindf_norm.iloc[:, 1:-1].to_numpy()
X_test = testdf_norm.iloc[:, 1:-1].to_numpy()
knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
knn.fit(X_train, y_train)
print('Score with normalized data: ' + str(knn.score(X_test, y_test)))

X_train = traindf_zscore_mod.iloc[:, :-2].to_numpy()
X_test = testdf_zscore_mod.iloc[:, :-2].to_numpy()
knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
knn.fit(X_train, y_train)
print('Score with weighted standardized data: ' + str(knn.score(X_test, y_test)))

X_train = traindf_norm_mod.iloc[:, :-2].to_numpy()
X_test = testdf_norm_mod.iloc[:, :-2].to_numpy()
knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
knn.fit(X_train, y_train)
print('Score with weighted normalized data: ' + str(knn.score(X_test, y_test)))

ok = 0
for h in range(len(testdf)):
    distance = []
    mindistance = []
    minindex = []
    for p in range(k):
        mindistance.append(1e5)
        minindex.append(0)
    for j in range(len(traindf_zscore)):
        distance.append(abs(traindf_zscore.iloc[j, -1] - testdf_zscore.iloc[h, -1]))
        for p in range(k):
            if distance[j] <= mindistance[p]:
                mindistance[p] = distance[j]
                minindex[p] = j
                break
    tot = 0
    for p in range(k):
        tot += traindf.iloc[minindex[p], traindf.columns.get_loc('cardio')]
    if tot >= (k // 2 + 1):
        testcardio = 1
    else:
        testcardio = 0
    if testcardio == testdf.iloc[h, testdf.columns.get_loc('cardio')]:
        ok += 1
    print(str(round(100 * ok / (h + 1), 1)) + ' %')
print('ok cases = ' + str(ok))
print('tot cases = ' + str(len(testdf.iloc[:, 1])))
print('accuracy = ' + str(ok / len(testdf.iloc[:, 1])))
