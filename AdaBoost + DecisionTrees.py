import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

save_dir = r"D:\啦啦啦啦啦啦啦我的东西\大学\大三下2026春\模式识别与机器学习\作业\HW2\Code"

train_data = np.loadtxt(os.path.join(save_dir, "train.csv"), delimiter=",", skiprows=1)
test_data = np.loadtxt(os.path.join(save_dir, "test.csv"), delimiter=",", skiprows=1)

X_train = train_data[:, :3]
y_train = train_data[:, 3]

X_test = test_data[:, :3]
y_test = test_data[:, 3]

base_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=9,
    random_state=42
)

ada_model = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)


ada_model.fit(X_train, y_train)

y_pred = ada_model.predict(X_test)


acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print("===== AdaBoost + Decision Tree Results =====")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["C0", "C1"]))
