import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

# Load and clean (mirrors the README's audit steps)
path = glob.glob("MachineLearningCVE/*PortScan*.csv")[0]
df = pd.read_csv(path)
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.duplicated()]                 # duplicate Fwd Header Length
df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates()
df["Label"] = df["Label"].str.strip()
print(df["Label"].value_counts())

y = (df["Label"] != "BENIGN").astype(int)                 # 1 = PortScan
X = df.drop(columns="Label")
X = X.loc[:, X.nunique() > 1]                             # drop constant features

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=0)

models = {
    "decision_tree": make_pipeline(SimpleImputer(strategy="median"),
                                   DecisionTreeClassifier(random_state=0)),
    "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                   RandomForestClassifier(n_jobs=-1, random_state=0)),
    "svm_rbf": make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(), SVC(kernel="rbf")),
}

for name, model in models.items():
    Xf, yf = X_tr, y_tr
    if name == "svm_rbf" and len(X_tr) > 50_000:          # SVM scales poorly; subsample
        Xf, _, yf, _ = train_test_split(X_tr, y_tr, train_size=50_000,
                                        stratify=y_tr, random_state=0)
    model.fit(Xf, yf)
    pred = model.predict(X_te)
    print(f"\n=== {name} ===  macro F1 = {f1_score(y_te, pred, average='macro'):.4f}")
    print(classification_report(y_te, pred, target_names=["BENIGN", "PortScan"], digits=4))
