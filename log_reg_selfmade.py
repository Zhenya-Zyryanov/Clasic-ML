import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from tqdm import tqdm
from tabulate import tabulate

np.set_printoptions(suppress=True, precision=4)
lambda_reg = 0.05


def data_pca(data_train, data_test, n_components=0.95):
    pca = PCA(n_components)
    dataframe_train_pca = pca.fit_transform(data_train)
    dataframe_test_pca = pca.transform(data_test)

    return dataframe_train_pca, dataframe_test_pca


def fit(x_data, y_data, k, max_iter=100000, learning_rate=0.1, eps=1e-4):
    n, p = x_data.shape
    M = np.zeros((n, k))
    M[np.arange(n), y_data] = 1

    B = np.zeros((k, p))
    b0 = np.zeros(k)
    print("Обучение модели:")
    for it in tqdm(range(max_iter)):
        S = (x_data @ B.T) + b0
        exp_S = np.exp(S - S.max(axis=1, keepdims=True))
        P = exp_S / exp_S.sum(axis=1, keepdims=True)
        E = M - P
        grad_B = - (E.T @ x_data) / n + lambda_reg * B
        grad_b0 = - E.sum(axis=0) / n

        B_new = B - learning_rate * grad_B
        b0_new = b0 - learning_rate * grad_b0

        if np.linalg.norm(B_new - B) < eps and np.linalg.norm(b0_new - b0) < eps:
            print(f"Обучение закончено на {it} итерации")
            break

        B = B_new
        b0 = b0_new

    return B, b0


def predict(x_data, B, b0):
    S = x_data @ B.T + b0
    exp_S = np.exp(S - S.max(axis=1, keepdims=True))
    P = exp_S / exp_S.sum(axis=1, keepdims=True)
    y_pred = np.argmax(P, axis=1)

    return y_pred, P


def predict_one(x_data, y_data, y_labels, B, b0):
    print(f"\nВыберите № строки из датасета [{0} ; {x_data.shape[0] - 1}]:")

    while True:
        try:
            choice = int(input("Ваш выбор = "))
            if 0 <= choice <= x_data.shape[0] - 1:
                break
            print(f"Введите целое число из интервала [{0} ; {x_data.shape[0] - 1}] !!")
        except ValueError:
            print("Введите целое число!!")

    x_choice = x_data[choice]
    S = x_choice @ B.T + b0
    exp_S = np.exp(S - S.max(keepdims=True))
    P = exp_S / exp_S.sum(keepdims=True)
    y_pred = np.argmax(P)

    P_reshaped = P.reshape(1, -1)
    predict_df = pd.DataFrame(P_reshaped, columns=y_labels)

    print(f"Предсказано = {y_labels[y_pred]} По факту = {y_data[choice]}")
    print("Вероятности:")
    print(tabulate(predict_df, headers="keys", tablefmt="grid", showindex=False))

    return None


def show_data(dataframe, answers=True, all=False):
    if (answers == True) and (all == False):
        print(tabulate(dataframe.head(), headers="keys", tablefmt="grid"))
    elif (answers == False) and (all == False):
        print(tabulate(dataframe.drop(["Activity", "subject"], axis=1).head(), headers="keys", tablefmt="grid"))
    elif (answers == True) and (all == True):
        print(tabulate(dataframe, headers="keys", tablefmt="grid"))
    elif (answers == False) and (all == True):
        print(tabulate(dataframe.drop(["Activity", "subject"], axis=1), headers="keys", tablefmt="grid"))

    return None


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("Пример как выглядят данные на которых обучается модель:")
show_data(train)
print("\nПример данных с которыми модель будет работать:")
show_data(test, answers=False)

x_train = train.drop(["Activity", "subject"], axis=1).to_numpy()
y_train = train["Activity"].to_numpy()
x_test = test.drop(["Activity", "subject"], axis=1).to_numpy()
y_test = test["Activity"].to_numpy()

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train, x_test = data_pca(x_train, x_test, n_components=0.95)
print(f"\nПосле PCA количество признаков = {x_train.shape[1]}")
x_train_pca_df = pd.DataFrame(x_train, columns=[f"PC{i+1}" for i in range(x_train.shape[1])])
x_test_pca_df = pd.DataFrame(x_test, columns=[f"PC{i+1}" for i in range(x_test.shape[1])])
show_data(x_test_pca_df)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

print(f"\nНаши классы: {le.classes_}")
k = len(le.classes_)
B, b0 = fit(x_train, y_train_enc, k)

y_pred, P_test = predict(x_test, B, b0)

y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test_enc)

y_pred_train, _ = predict(x_train, B, b0)
y_pred_test, _ = predict(x_test, B, b0)

metrics = calculate_metrics(
    y_train_enc, y_pred_train,
    y_test_enc, y_pred_test,
    y_test_labels, y_pred_labels,
    P_test, le.classes_
)


"""# Небольшой пример работы
for _ in range(3):
    predict_one(x_test_har, y_test_labels, le.classes_, B, b0)
"""


