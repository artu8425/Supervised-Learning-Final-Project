import pandas as pd

df = pd.read_csv("student-mat.csv", sep=';')

print(df.shape)
print(df.head())
print(df.info())

df["pass"] = (df["G3"] >= 10).astype(int)

df_encoded = pd.get_dummies(df.drop(columns=["G1","G2","G3"]), drop_first=True)
print(df_encoded.shape)

from sklearn.model_selection import train_test_split

X = df_encoded.drop(columns=["pass"])
y = df_encoded["pass"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Logistic Regression
log = LogisticRegression(max_iter=1000, random_state=42)
log.fit(X_train, y_train)
print("Logistic accuracy:", accuracy_score(y_test, log.predict(X_test)))

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest accuracy:", accuracy_score(y_test, rf.predict(X_test)))

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(2), yticks=np.arange(2),
    xticklabels=["Fail","Pass"], yticklabels=["Fail","Pass"],
    xlabel="Predicted label", ylabel="True label",
    title="Random Forest Confusion Matrix"
)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")
plt.show()