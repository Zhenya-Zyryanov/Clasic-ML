from Softmax_model.Softmax import SoftmaxClassifier
from Decision_Trees_model.decision_trees import DecisionTree_classification
from Decision_Trees_model.decision_trees import RandomForest_classification
from Decision_Trees_model.decision_trees import GBDT_classification
from Decision_Trees_model.decision_trees import GBDT_with_sklearn_classification
from Neural_Network_model.MLP_network import MLP_network
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from joblib import dump, load
import pandas as pd
import os


# Загрузка данных
har_train = pd.read_csv('Datasets/har_train.csv')
har_test = pd.read_csv('Datasets/har_test.csv')
x_train_har = har_train.drop(['Activity', 'subject'], axis=1).to_numpy()
y_train_har = har_train['Activity'].to_numpy()
x_test_har = har_test.drop(['Activity', 'subject'], axis=1).to_numpy()
y_test_har = har_test['Activity'].to_numpy()

# логрег
print("========== Softmax work ==========")
softmax_file = "softmax_model.joblib"

if softmax_file in os.listdir("Saved models"):
    softmax_model = load(softmax_file)
else:
    softmax_model = SoftmaxClassifier(
        learning_rate=0.1,
        max_iter=100000,
        eps=1e-4,
        lambda_reg=0.05,
        use_pca=True,
        n_components=0.95
    )
    softmax_model.fit(x_train_har, y_train_har)
    dump(softmax_model, "Softmax_model/softmax_model.joblib", )

# Предсказание и оценка
softmax_pred_test = softmax_model.predict(x_test_har)
softmax_pred_train = softmax_model.predict(x_train_har)
# Преобразование меток в числовые
le = LabelEncoder()
y_train_enc = le.fit(y_train_har)
softmax_pred_train = le.inverse_transform(softmax_pred_train)
softmax_pred_test = le.inverse_transform(softmax_pred_test)

# Метрики
print(f"Accuracy: on har_train = {accuracy_score(y_train_har, softmax_pred_train)}")
print(f"Accuracy: on har_test = {accuracy_score(y_test_har, softmax_pred_test)}")
print(f"Confusion matrix:\n{confusion_matrix(y_test_har, softmax_pred_test)}")
print(f"Classification report:\n{classification_report(y_test_har, softmax_pred_test)}")

# Небольшой пример работы
"""for _ in range(4):
    model.predict_one(x_test_har, y_test_har)
"""

print("\n========== Tree work ==========")
tree_file = "tree_classification_model.joblib"
if tree_file in os.listdir("Saved models"):
    tree = load(tree_file)
else:
    tree = DecisionTree_classification(max_depth=16)
    tree.fit(x_train_har, y_train_har)
    dump(tree, "Saved models/tree_classification_model.joblib")

tree_pred = tree.predict(x_test_har)

print(f"Accuracy: {accuracy_score(y_test_har, tree_pred)}")
print(f"Confusion Matrix: {confusion_matrix(y_test_har, tree_pred)}")
print(f"Classification Report: {classification_report(y_test_har, tree_pred)}")


print("\n========== Forest work ==========")
forest_file = "forest_classification_model.joblib"
if forest_file in os.listdir("Saved models"):
    forest = load(forest_file)
else:
    forest = RandomForest_classification(max_depth=16, n_trees=50, method="sqrt")
    forest.fit(x_train_har, y_train_har)
    dump(forest, "Saved models/forest_classification_model.joblib")

forest_pred = forest.predict(x_test_har)

print(f"Accuracy: \n{accuracy_score(y_test_har, forest_pred)}")
print(f"Confusion Matrix: \n{confusion_matrix(y_test_har, forest_pred)}")
print(f"Classification Report: \n{classification_report(y_test_har, forest_pred)}")

print("\n========== Gradient boosting decision tree work ==========")
GBDT_file = "GBDT_classification_model.joblib"

if GBDT_file in os.listdir("Saved models"):
    gbdt_model = load(GBDT_file)
else:
    gbdt_model = GBDT_classification(n_estimators=50, max_depth=6, lambda_reg=0.05, alpha_reg=0.05)
    gbdt_model.fit(x_train_har, y_train_har)
    dump(gbdt_model, "Saved models/GBDT_classification_model.joblib")

gbdt_pred_train = gbdt_model.predict(x_train_har)
gbdt_pred_test = gbdt_model.predict(x_train_har)

print(f"Accuracy: on har_train = {accuracy_score(y_test_har, gbdt_pred_train)}")
print(f"Accuracy: on har_test = {accuracy_score(y_test_har, gbdt_pred_test)}")
print(f"Confusion Matrix: \n{confusion_matrix(y_test_har, gbdt_pred_test)}")
print(f"Classification Report: \n{classification_report(y_test_har, gbdt_pred_test)}")

print("\n========== MLP neural network work ==========")
