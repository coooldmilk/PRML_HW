import os
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

save_dir = r"D:\啦啦啦啦啦啦啦我的东西\大学\大三下2026春\模式识别与机器学习\作业\HW2\Code"

train_data = np.loadtxt(os.path.join(save_dir, "train.csv"), delimiter=",", skiprows=1)
test_data = np.loadtxt(os.path.join(save_dir, "test.csv"), delimiter=",", skiprows=1)

X_train = train_data[:, :3]
y_train = train_data[:, 3]

X_test = test_data[:, :3]
y_test = test_data[:, 3]

svm_models = {
    """ 
    统一 C=1.0 C->对“分类错误”的容忍程度
    kernel="poly" 多项式核
    degree=3 表示三次多项式核
    RBF 高斯核 SVM

    make_pipeline
    在训练集上拟合标准化器
    把训练集变换
    训练 SVM
    预测测试集时，用同一个标准化器去变换测试集
    """

    "SVM_Linear": make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=1.0)
    ),
    "SVM_Polynomial": make_pipeline(
        StandardScaler(),
        SVC(kernel="poly", degree=3, C=1.0, gamma="scale")
    ),
    "SVM_RBF": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale")
    ),
    "SVM_Sigmoid": make_pipeline(
        StandardScaler(),
        SVC(kernel="sigmoid", C=1.0, gamma="scale")
    )
}


for model_name, model in svm_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {model_name} Results =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["C0", "C1"]))
