import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from tqdm import tqdm


class SoftmaxClassifier:
    def __init__(
        self,
        learning_rate=0.1,
        max_iter=100000,
        eps=1e-4,
        lambda_reg=0.05,
        use_pca=False,
        n_components=0.95,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.lambda_reg = lambda_reg
        self.use_pca = use_pca
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components) if use_pca else None
        self.encoder = LabelEncoder()
        self.B = None
        self.b0 = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        y_enc = self.encoder.fit_transform(y_train)
        num_classes = len(self.encoder.classes_)

        if self.use_pca:
            X = self.scaler.fit_transform(x_train)
            X = self.pca.fit_transform(X)
        else:
            X = self.scaler.fit_transform(x_train)

        n_samples, n_features = X.shape
        self.B = np.zeros((num_classes, n_features))
        self.b0 = np.zeros(num_classes)

        M = np.zeros((n_samples, num_classes))
        M[np.arange(n_samples), y_enc] = 1

        print("Обучение модели:")
        for _ in tqdm(range(self.max_iter)):
            S = X @ self.B.T + self.b0
            exp_S = np.exp(S - S.max(axis=1, keepdims=True))
            P = exp_S / exp_S.sum(axis=1, keepdims=True)

            E = M - P
            grad_B = -(E.T @ X) / n_samples + self.lambda_reg * self.B
            grad_b0 = -E.sum(axis=0) / n_samples

            B_new = self.B - self.learning_rate * grad_B
            b0_new = self.b0 - self.learning_rate * grad_b0

            if (np.linalg.norm(self.B - B_new) < self.eps and
                np.linalg.norm(self.b0 - b0_new) < self.eps):
                self.B, self.b0 = B_new, b0_new
                break

            self.B, self.b0 = B_new, b0_new

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.use_pca:
            X_scaled = self.scaler.transform(X)
            X_scaled = self.pca.transform(X_scaled)
        else:
            X_scaled = self.scaler.transform(X)

        S = X_scaled @ self.B.T + self.b0
        exp_S = np.exp(S - S.max(axis=1, keepdims=True))
        return exp_S / exp_S.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.use_pca:
            X_scaled = self.scaler.transform(X)
            X_scaled = self.pca.transform(X_scaled)
        else:
            X_scaled = self.scaler.transform(X)

        S = X_scaled @ self.B.T + self.b0
        exp_S = np.exp(S - S.max(axis=1, keepdims=True))
        P = exp_S / exp_S.sum(axis=1, keepdims=True)
        return np.argmax(P, axis=1)
