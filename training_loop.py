import numpy as np
from sklearn.utils import shuffle

class AdamOptimizer:
    def __init__(self, params_shape, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(params_shape)
        self.v = np.zeros(params_shape)
        self.t = 0

    def step(self, weights, grads):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        updated_weights = weights - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return updated_weights

# qnn obiekt z klasy od marysi
# X_train, y_train dane do trenowania od kariny
# initial_weights wagi startowe od marysi
#loss_history = [] do wykresow na podsumowanie do iwo 

EPOCHS = 20 
BATCH_SIZE = 32
LEARNING_RATE = 0.02

weights = initial_weights.copy()
loss_history = []

optimizer = AdamOptimizer(weights.shape, lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=epoch)
    
    epoch_loss = 0.0
    batches_count = 0
    
    for i in range(0, len(X_train), BATCH_SIZE):
        X_batch = X_train_shuffled[i:i + BATCH_SIZE]
        y_batch = y_train_shuffled[i:i + BATCH_SIZE]
        
        pred = qnn.forward(X_batch, weights)
        _, grads = qnn.backward(X_batch, weights)

        diff = pred - y_batch.reshape(-1, 1)
        loss = np.mean(diff ** 2)

        grad_modifier = 2 * diff

        batch_grads = np.mean(grad_modifier[:, :, np.newaxis] * grads, axis=0).squeeze()

        weights = optimizer.step(weights, batch_grads)
        
        epoch_loss += loss
        batches_count += 1
    
    avg_loss = epoch_loss / batches_count
    loss_history.append(avg_loss) 
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg loss: {avg_loss:.4f}")