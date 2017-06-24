import pandas as pd
import numpy as np
from decimal import *
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn import preprocessing, linear_model, svm
from pprint import pprint
import seaborn as sns
import cntk as C
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

input_dim = 3

PRECISION = 15
sns.set(color_codes=True)
from scipy import stats, integrate

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.feature_extraction import DictVectorizer




pd.options.display.max_seq_items = 20000
dt = pd.DataFrame.from_csv('TX_I-20_RawData.csv')

print(dt.columns.values[2])


def plot(maxIndex,importanceFeature):
    for i in range(1,maxIndex):
     popValue = importanceFeature[i]
     print(popValue)
     sns.distplot(X.iloc[:,popValue])
     sns.plt.show()

# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss, eval_error = "NA", "NA"

    if mb % frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}".format(mb, training_loss, eval_error))

    return mb, training_loss, eval_error


#print(dt.corr())
save = dt.corr().abs()
s=save.unstack()
so = s.sort_values(kind="quicksort",ascending=False)
so = round(so,2)
np.around(so, decimals=2)
#print(so)
le = preprocessing.LabelEncoder()

clf = LassoCV()
y=dt['Deaths by CVD'].values
X = dt.ix[:,0:70]


X['Areaname'] = X['Areaname'].astype('category')
X['Areaname'] = X['Areaname'].cat.reorder_categories(['Reeves, TX','Ward, TX','Crane, TX','Ector, TX','Midland, TX','Martin, TX','Howard, TX','Mitchell, TX','Nolan, TX','Taylor, TX','Callahan, TX','Eastland, TX','Erath, TX','Palo Pinto, TX','Parker, TX','Tarrant, TX','Dallas, TX','Kaufman, TX','Van Zandt, TX','Smith, TX','Gregg, TX','Harrison, TX'], ordered=True)
X['Areaname'] = X['Areaname'].cat.codes


model = LogisticRegression()
rfe = RFE(model, 10)
fit = rfe.fit(X, y)



# print("Num Features: ",fit.n_features_)
# print("Selected Features: ",fit.support_)
# print("Feature Ranking: ",fit.ranking_)

#pca = PCA(n_components=3)
#fit = pca.fit(X)

model = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
model.fit(X, y)
matrix = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(matrix)[::-1]
# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], matrix[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), matrix[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()




# matrix = []
# notSorted=[]
#
# notSorted = np.around(matrix[:], decimals=PRECISION)
#
# #Get importance feature
# matrix=sorted(matrix, reverse = True)
# matrix=np.around(matrix, decimals=PRECISION)
#
# output = []
# index = []

#Print the importance feature

# for j, item in enumerate(matrix):
#       for i, item2 in enumerate(notSorted):
#           if np.isclose(item2, item, 0.01) & (item2!=0):
#               output.append(i)
#               #print('Index in old array is :',i, ' in sorted array',j )
#               #print(i, end=' ')

#print(output)

# print(matrix,end=' ')
# for i,item in enumerate(notSorted):
#     print (i,".",item,"\t",dt.columns.values[i])


#plot(20,indices)



listOfHeaders= []
for f in range(1, input_dim):
    #listOfHeaders.append(dt.columns.values[indices[f]])
    listOfHeaders.append(dt.columns.values[indices[f]])
    #data.append(dt.ix[:, indices[f]].values)
    #training.append(dt.ix[:,indices[f]])

print(listOfHeaders)
# Create linear regression object
#regr = linear_model.LinearRegression()
regr = linear_model.LinearRegression()

# Train the model using the training sets
#print('target',y)
#data=list(zip(*data))


data=dt[listOfHeaders].values


print ("Data : ",data)
print ("Target :",y)

train_data, test_data, traing_y, testing_y = train_test_split(data, y, test_size=0.20, random_state=42)

print(train_data)
#print(len(test_data))
print(traing_y)
# print(len(testing_y))

# print('Training row', len(data))
# print('Target row',len(y))

regr.fit(np.array(train_data), np.array(traing_y))

# The coefficients
#print('Coefficients: \n', regr.coef_)


print("Mean squared error: %.2f"% mean_squared_error(testing_y, regr.predict(test_data)))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(test_data, testing_y))