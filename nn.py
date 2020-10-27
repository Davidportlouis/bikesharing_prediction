import numpy as np

class NeuralNet:
    def __init__(self, in_features, hidden_features, out_features, alpha):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.alpha = alpha
        self.weights_input_to_hidden = np.random.normal(0.0, in_features**-0.5,
                (self.in_features, self.hidden_features))
        self.weights_hidden_to_output = np.random.normal(0.0, hidden_features**-0.5,
                (self.hidden_features, self.out_features))
        self.activation_function = lambda x: 1.0/(1.0 + np.exp(-x))


    def train(self, features, targets):
        m, n = features.shape
        delta_weights_i_h = np.zeros_like(self.weights_input_to_hidden)
        delta_weights_h_o = np.zeros_like(self.weights_hidden_to_output)
        for X, y in zip(features, targets):
            y_preds, hidden_outputs = self.forward(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagate(y_preds, hidden_outputs, X, y,
                    delta_weights_i_h, delta_weights_h_o)
        self.SGD(delta_weights_i_h, delta_weights_h_o, m)


    def forward(self, X):
        z_1 = np.dot(X,self.weights_input_to_hidden)
        h_1 = self.activation_function(z_1)
        z_2 = np.dot(h_1,self.weights_hidden_to_output)
        h_2 = 1 * z_2
        return h_2, h_1


    def backpropagate(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        error = y - final_outputs
        hidden_error = np.dot(error,self.weights_hidden_to_output.T)
        output_error_term = error
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        delta_weights_i_h += hidden_error_term * X[:, None]
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        return  delta_weights_i_h, delta_weights_h_o


    def SGD(self, delta_weights_i_h, delta_weights_h_o, n_records):
        self.weights_input_to_hidden += self.alpha * delta_weights_i_h / n_records
        self.weights_hidden_to_output += self.alpha * delta_weights_h_o / n_records

    
    def predict(self, features):
        z_1 = np.dot(features,self.weights_input_to_hidden)
        h_1 = self.activation_function(z_1)
        z_2 = np.dot(h_1,self.weights_hidden_to_output)
        h_2 = 1 * z_2
        return h_2


