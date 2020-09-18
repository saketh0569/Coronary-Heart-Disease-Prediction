import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

# reading data
data_CHD = pd.read_csv("framingham.csv")
data_CHD.drop(['education'], axis=1, inplace=True)
print(data_CHD.head(10))
print(data_CHD.info())

# Analysing Data

# total percentage of missing data
missing_data = data_CHD.isnull().sum()
total_percentage = (missing_data.sum() / data_CHD.shape[0]) * 100
print(f'The total percentage of missing data is {round(total_percentage, 2)}%')

# percentage of missing data per category
total = data_CHD.isnull().sum().sort_values(ascending=False)
percent_total = (data_CHD.isnull().sum() / data_CHD.isnull().count()).sort_values(ascending=False) * 100
missing = pd.concat([total, percent_total], axis=1, keys=["Total", "Percentage"])
missing_data = missing[missing['Total'] > 0]
print(missing_data)

# plotting percentage of missing data
plt.figure(figsize=(9, 6))
sns.set(style="whitegrid")
sns.barplot(x=missing_data.index, y=missing_data['Percentage'], data=missing_data)
plt.title('Percentage of missing data by feature')
plt.xlabel('Features', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.savefig("Per_of_missing data_by_feature")

# plotting histogram of all variables
fig = plt.figure(figsize=(15, 20))
ax = fig.gca()
data_CHD.hist(ax=ax)
plt.savefig("Histogram_variables")

# Missing Values Treatment

data_CHD.dropna(axis=0, inplace=True)

# plotting histogram of output variable
plt.figure()
fig = plt.figure(figsize=(9, 6))
sns.countplot(x='TenYearCHD', data=data_CHD)
plt.savefig("CHD_Histogram")
cases = data_CHD.TenYearCHD.value_counts()
print(f"There are {cases[0]} patients without heart disease and {cases[1]} patients with the disease")

# no of people disease vs age
plt.figure(figsize=(15, 6))
sns.countplot(x='age', data=data_CHD, hue='TenYearCHD', palette='husl')
plt.savefig("disease_vs_age")
print("The people with the highest risk of developing CHD are between the ages of 51 and 63 ")

# Correlation heat map
plt.figure(figsize=(15, 8))
sns.heatmap(data_CHD.corr(), annot=True)
plt.savefig("Heat_map")

# There are no features with more than 0.5 correlation with the Ten year risk of developing CHD.

# Also there are a couple of features that are highly correlated with one another.These includes:
#  systolic and diastolic blood pressures
#  cigarette smoking and the number of cigarettes smoked per day.
#  Blood glucose and diabetes


# outlier detection
fig, ax = plt.subplots(figsize=(10, 10), nrows=3, ncols=4)
ax = ax.flatten()

i = 0
for k, v in data_CHD.items():
    sns.boxplot(y=v, ax=ax[i])
    i += 1
    if i == 12:
        break
plt.tight_layout(pad=1.25, h_pad=0.8, w_pad=0.8)
plt.savefig("Boxplot")

# Conclusion of Boxplot :
# Outliers found in features named ['totChol', 'sysBP', 'BMI','heartRate', 'glucose']

# remove outliers

data_CHD = data_CHD[~(data_CHD['diaBP'] > 130)]
data_CHD = data_CHD[~(data_CHD['BMI'] > 43)]
data_CHD = data_CHD[~(data_CHD['heartRate'] > 125)]
data_CHD = data_CHD[~(data_CHD['glucose'] > 200)]
data_CHD = data_CHD[~(data_CHD['totChol'] > 450)]

# feature selection
y = data_CHD['TenYearCHD']
X = data_CHD.drop('TenYearCHD', axis=1)
CHD = list(data_CHD['TenYearCHD'])
smoke = list(data_CHD['currentSmoker'])
smoke_CHD = int(0)
nonSmoke_CHD = int(0)

for i in range(len(smoke)):
    if smoke[i] == 1 and CHD[i] == 1:
        smoke_CHD = smoke_CHD + 1
    elif smoke[i] == 0 and CHD[i] == 1:
        nonSmoke_CHD = nonSmoke_CHD + 1
no_of_smokers = int(0)
for a in smoke:
    if a == 1:
        no_of_smokers = no_of_smokers + 1
no_of_nonsmokers = len(smoke) - no_of_smokers
print("Smokers with CHD" + str(smoke_CHD / no_of_smokers))
print("non_Smokers with CHD" + str(nonSmoke_CHD / no_of_nonsmokers))
num_before = dict(Counter(y))

X = data_CHD.iloc[:, :-1].values
y = data_CHD.iloc[:, -1].values

forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced')

# define Boruta feature selection method
feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2)

# find all relevant features
feat_selector.fit(X, y)
top_features = data_CHD.columns[:-1][feat_selector.ranking_ <= 6].tolist()
print(top_features)

# data balancing

X = data_CHD[top_features]
y = data_CHD.iloc[:, -1]
# the numbers before smote
num_before = dict(Counter(y))

# perform smoting

# define pipeline
over = SMOTE(sampling_strategy=0.8)
under = RandomUnderSampler(sampling_strategy=0.8)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# transform the dataset
X_smote, y_smote = pipeline.fit_resample(X, y)

# the numbers after smote

num_after = dict(Counter(y_smote))
print(num_before, num_after)
labels = ["Negative Cases", "Positive Cases"]
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.barplot(labels, list(num_before.values()))
plt.title("Numbers Before Balancing")
plt.subplot(1, 2, 2)
sns.barplot(labels, list(num_after.values()))
plt.title("Numbers After Balancing")
plt.savefig("After_balancing")
new_data = pd.concat([pd.DataFrame(X_smote), pd.DataFrame(y_smote)], axis=1)
new_data.columns = ['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']
# new_data_set
X_new = new_data[top_features]
print(X_new.head(5))
y_new = new_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=.2, random_state=42)

# scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_scaled)

# Neural Networks

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model = ann.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=100)

ann_y_pred = ann.predict(X_test)
ann_y_pred = ann_y_pred > 0.5
cm = confusion_matrix(y_test, ann_y_pred)
ann_f1 = f1_score(y_test, ann_y_pred)
print(cm)
ann_accuracy = accuracy_score(y_test, ann_y_pred)
ann_acc = round(ann_accuracy * 100, 2)
print(accuracy_score(y_test, ann_y_pred))
print(classification_report(y_test, ann_y_pred))

# decision tree
dtree = DecisionTreeClassifier(random_state=7)
params = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
          'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
tree_clf = GridSearchCV(dtree, param_grid=params, n_jobs=-1)
# train the model
tree_clf.fit(X_train, y_train)
# predictions
tree_predict = tree_clf.predict(X_test)
# accuracy
tree_accuracy = accuracy_score(y_test, tree_predict)
tree_acc = round(tree_accuracy * 100, 2)
cm = confusion_matrix(y_test, tree_predict)
dtree_f1 = f1_score(y_test, tree_predict)
print(f"Using Decision Trees we get an accuracy of {round(tree_accuracy * 100, 2)}%")
print(classification_report(y_test, tree_predict))

# KNN
params = {'n_neighbors': np.arange(1, 10)}
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params,
                           scoring='accuracy', cv=10, n_jobs=-1)
knn_clf = GridSearchCV(KNeighborsClassifier(), params, cv=3, n_jobs=-1)
knn_clf.fit(X_train, y_train)
print(knn_clf.best_params_)
knn_predict = knn_clf.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predict)
knn_acc = round(knn_accuracy * 100, 2)
knn_f1 = f1_score(y_test, knn_predict)
print(f"Using k-nearest neighbours we get an accuracy of {round(knn_accuracy * 100, 2)}%")
print(classification_report(y_test, knn_predict))

# xgboost ensemble
model = XGBClassifier(max_depth=12, subsample=1,
                      n_estimators=1250, learning_rate=.090,
                      min_child_weight=1, random_state=10)
model.fit(X_train, y_train)
y_preds = model.predict(X_test)
y_train_preds = model.predict(X_train)
xgb_accuracy = accuracy_score(y_test, y_preds)
xgb_f1 = f1_score(y_test, y_preds)
xgb_acc = round(accuracy_score(y_test, y_preds)*100, 2)
print(f"Accuracy of xgb classifier is {round(accuracy_score(y_test, y_preds)*100, 2)}%")
print(classification_report(y_test, y_preds))

comparison = pd.DataFrame({
    "ANN ": {'Accuracy': ann_acc, 'F1 score': ann_f1},
    "D-Tree": {'Accuracy': tree_acc, 'F1 score': dtree_f1},
    "K-nearest neighbours": {'Accuracy': knn_acc, 'F1 score': knn_f1},
    "XGBoost": {'Accuracy': xgb_acc, 'F1 score': xgb_f1}
}).T
print(comparison)