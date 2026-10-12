import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

TEST_SIZE = 0.4

dataset = pd.read_csv('MachineLearningCSV/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                      low_memory=True)
dataset.columns = dataset.columns.str.strip()

# stratified 100,000-flow sample, as in the repo, so the SVM finishes in reasonable time
label = dataset.columns[-1]
_, dataset = train_test_split(dataset, test_size=100000, random_state=0,
                              stratify=dataset[label])
print(dataset[label].value_counts())

X = dataset.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = SimpleImputer(strategy='mean').fit_transform(X)
y = LabelEncoder().fit_transform(dataset.iloc[:, -1])      # BENIGN=0, PortScan=1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=0, stratify=y)

models = {
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
    # raw flow features need scaling for an RBF SVM (the repo's SVM gets LDA output instead)
    "SVM": make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=0)),
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n===== {name} =====")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
    print("Precision: {:.4f}".format(precision_score(y_test, y_pred, average='weighted')))
    print("Recall   : {:.4f}".format(recall_score(y_test, y_pred, average='weighted')))
    print("F1 Score : {:.4f}".format(f1_score(y_test, y_pred, average='weighted')))
    print(classification_report(y_test, y_pred, target_names=["BENIGN", "PortScan"]))
