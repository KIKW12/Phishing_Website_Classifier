import numpy as np
import pandas as pd

training_data = pd.read_csv('df_train.csv')
test_data = pd.read_csv('df_test.csv')

class NN(object):
    def __init__(self):
        self.input = 10
        self.output = 1
        self.hidden_units = 11
        
        np.random.seed(1)
        self.w1 = np.random.randn(self.input, self.hidden_units) * 0.01  
        self.w2 = np.random.randn(self.hidden_units, self.output) * 0.01  
    
    def save_model(self, w1_file='w1.npy', w2_file='w2.npy'):
        np.save(w1_file, self.w1)
        np.save(w2_file, self.w2)
        print("Model weights saved successfully.")

    def load_model(self, w1_file='w1.npy', w2_file='w2.npy'):
        self.w1 = np.load(w1_file)
        self.w2 = np.load(w2_file)
        print("Model weights loaded successfully.")

    def _forward_propagation(self, X):
        self.z2 = np.dot(X, self.w1)  
        self.a2 = self._sigmoid(self.z2)  
        self.z3 = np.dot(self.a2, self.w2)  
        self.a3 = self._sigmoid(self.z3)  
        return self.a3

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)  
        return 1 / (1 + np.exp(-z))

    def _loss(self, predict, y):
        m = y.shape[0]
        logprobs = y * np.log(predict) + (1 - y) * np.log(1 - predict)
        loss = -np.sum(logprobs) / m
        return loss

    def _backward_propagation(self, X, y):
        predict = self.a3  
        m = X.shape[0]
        
        
        delta3 = predict - y
        self.dw2 = np.dot(self.a2.T, delta3) / m  
        db2 = np.sum(delta3, axis=0) / m  

        
        delta2 = np.dot(delta3, self.w2.T) * self._sigmoid_prime(self.z2)
        self.dw1 = np.dot(X.T, delta2) / m  
        db1 = np.sum(delta2, axis=0) / m  

        
        self.w1 -= self.learning_rate * self.dw1
        self.w2 -= self.learning_rate * self.dw2

    def _sigmoid_prime(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))
    
    def train(self, X, y, iterations=33, learning_rate=1.2):
        self.learning_rate = learning_rate
        y = y.values.reshape(-1, 1)  

        for i in range(iterations):
            y_hat = self._forward_propagation(X)
            loss = self._loss(y_hat, y)
            self._backward_propagation(X, y)

            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, X):
        y_hat = self._forward_propagation(X)
        y_hat = (y_hat >= 0.5).astype(int)  
        return y_hat

    def score(self, y_pred, y_true):
        return (np.mean(y_pred == y_true.values.reshape(-1, 1))) * 100

if __name__=='__main__':
    train_X = training_data[['use_of_ip', 'count.', '@ Precence', '- Precence', '∼ Precence', 'count_embed_domian', 'sus_url', 'short_url', 'HTTPS in Domain', 'url_length']]
    test_X =  test_data[['use_of_ip', 'count.', '@ Precence', '- Precence', '∼ Precence', 'count_embed_domian', 'sus_url', 'short_url', 'HTTPS in Domain', 'url_length']]
    train_y = training_data['label_code']
    test_y = test_data['label_code']
    
    clr = NN()
    clr.train(train_X, train_y, iterations=100)
    pre_y = clr.predict(test_X)
    score = clr.score(pre_y, test_y)
    print("Predictions:", pre_y.flatten())
    print("Actual:", test_y.values)
    print("Accuracy Score:", score)
    clr.save_model()