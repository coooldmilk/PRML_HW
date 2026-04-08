import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

save_dir = r"D:\啦啦啦啦啦啦啦我的东西\大学\大三下2026春\模式识别与机器学习\作业\HW2\Code"

train_data = np.loadtxt(os.path.join(save_dir, "train.csv"), delimiter=",", skiprows=1)
test_data = np.loadtxt(os.path.join(save_dir, "test.csv"), delimiter=",", skiprows=1)

X_train = train_data[:, :3] #1.24, -0.53, 0.88, 0/1 feature
y_train = train_data[:, 3]  #label


X_test = test_data[:, :3]
y_test = test_data[:, 3]


dt_model = DecisionTreeClassifier(
    criterion="gini", #Gini impurity 基尼不纯度 lower
    max_depth=9, #最大深度
    random_state=42
)

#学习+预测
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test) #[0, 0, 1, 1, 0, 1, ...]

""" 
评价指标
Accuracy 整体正确率
F1-score 类别平衡Precision Recall
Confusion Matrix 混淆矩阵
"""
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print("===== Decision Tree Results =====")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["C0", "C1"]))
