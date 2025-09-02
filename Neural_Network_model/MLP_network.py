import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class MLP_network:
    def __init__(
            self,
            hidden_sizes=None,
            activation_fanction='relu',
            lambda_reg=0.0,
            learning_rate=0.01,
            eps=1e-4,
            max_epochs=1000
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64, 32]
        self.hidden_sizes = hidden_sizes
        self.activation_fanction = activation_fanction
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.eps = eps
        self.max_epochs = max_epochs

        self.layer_sizes = None
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.classes_ = None

    def _activation_fanction(self, x):
        if self.activation_fanction == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_fanction == 'relu':
            return np.maximum(0, x)
        elif self.activation_fanction == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"Неподходящая функция активации: {self.activation_fanction}")

    def _activation_function_deriv(self, x):
        if self.activation_fanction == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif self.activation_fanction == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fanction == 'tanh':
            t = np.tanh(x)
            return 1 - t**2
        else:
            raise ValueError(f"Неподходящая функция активации: {self.activation_fanction}")

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _loss_function(self, Y_true, Y_pred):
        n = Y_true.shape[0]
        eps = 1e-15
        ce = -np.sum(Y_true * np.log(Y_pred + eps)) / n
        l2 = 0.0
        for W in self.weights:
            l2 += 0.5 * self.lambda_reg * np.sum(W**2)
        return ce + l2

    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        K = len(self.classes_)

        Y = np.zeros((n_samples, K))
        Y[np.arange(n_samples), y_idx] = 1

        self.layer_sizes = [n_features] + self.hidden_sizes + [K]
        L = len(self.layer_sizes) - 1

        self.weights = []
        self.biases = []
        for l in range(L):
            fan_in = self.layer_sizes[l]
            fan_out = self.layer_sizes[l+1]
            if self.activation_fanction == 'relu' and l < L-1:
                std = np.sqrt(2.0 / fan_in)
            else:
                fan_avg = 0.5 * (fan_in + fan_out)
                std = np.sqrt(1.0 / fan_avg)
            W = np.random.normal(0, std, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

        prev_loss = np.inf
        self.loss_history = []
        for epoch in tqdm(range(self.max_epochs), desc="Обучение MLP"):
            A = X
            activations = [A]
            pre_acts = []
            for l in range(L):
                Z = A @ self.weights[l] + self.biases[l]
                pre_acts.append(Z)
                if l < L-1:
                    A = self._activation_fanction(Z)
                else:
                    A = self._softmax(Z)
                activations.append(A)

            loss = self._loss_function(Y, activations[-1])
            self.loss_history.append(loss)
            if verbose and epoch % 200 == 0:
                print(f"Эпоха {epoch}, loss={loss:.6f}")

            if abs(prev_loss - loss) < self.eps:
                if verbose:
                    print(f"Остановка на {epoch} эпохе, change={abs(prev_loss-loss):.2e} < eps={self.eps}")
                break
            prev_loss = loss

            delta = activations[-1] - Y
            grads_W = [None]*L
            grads_b = [None]*L
            for l in range(L-1, -1, -1):
                A_prev = activations[l]
                grads_W[l] = (A_prev.T @ delta) / n_samples + self.lambda_reg * self.weights[l]
                grads_b[l] = np.mean(delta, axis=0, keepdims=True)
                if l > 0:
                    delta = (delta @ self.weights[l].T) * self._activation_function_deriv(pre_acts[l-1])

            for l in range(L):
                self.weights[l] -= self.learning_rate * grads_W[l]
                self.biases[l]  -= self.learning_rate * grads_b[l]

        return self

    def predict(self, X):
        A = X
        for l in range(len(self.weights)):
            Z = A @ self.weights[l] + self.biases[l]
            if l < len(self.weights)-1:
                A = self._activation_fanction(Z)
            else:
                A = self._softmax(Z)
        idx = np.argmax(A, axis=1)
        return self.classes_[idx]