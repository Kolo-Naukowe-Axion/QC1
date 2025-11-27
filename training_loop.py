import numpy as np
from sklearn.utils import shuffle

#zakladam ze wyzej sa juz zdefiniowane te zmienne:
#qnn, X_train, y_train, initial_weights (ewentulanie moge dodac generowanie losowych wag), AdamOptimizer
#loss_history = [] do wykresow na podsumowanie 

#hiperparametry do ustawienia
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