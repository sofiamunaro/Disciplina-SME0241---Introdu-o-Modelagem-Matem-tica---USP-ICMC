#rede mlp

import numpy as np

np.random.seed(7)
LEARNING_RATE = 0.01
EPOCHS = 20

#                x1            x2          x3     classe
dataset = [[ 1.02153588, -1.04584554, -0.96943922,  0], 
           [ 1.07472359, -0.81372418, -0.50227571,  0], 
           [ 1.03845087, -0.85529440, -1.07551718,  0], 
           [ 1.08671323, -0.39041877, -1.06912290,  0], 
           [ 1.81386050, -1.03705351, -1.01491796,  0], 
           [ 1.60285424, -0.53666876, -0.68868644,  0], 
           [ 1.67743150, -0.68302534, -0.58474262,  0], 
           [ 1.34016832, -0.33036141, -0.45972834,  0], 
           [ 1.15715279, -0.57687815, -0.86905314,  0], 
           [ 1.19312552, -0.74858765, -0.26682942,  0], 
           [ 1.32535135, -0.51619190, -0.50434856,  0], 
           [ 1.17080360, -0.29582611, -0.31267014,  0], 
           [-0.60098243,  1.13155099, -0.77290891,  1], 
           [-0.78660704,  1.22050116, -1.05339234,  1], 
           [-0.42366120,  1.29704384, -0.94234171,  1], 
           [-0.66885149,  1.43052978, -1.00206341,  1], 
           [-0.46490586,  1.76682434, -0.74631286,  1], 
           [-1.02489359,  1.35547338, -0.38458331,  1], 
           [-0.96717853,  1.46557232, -0.68402367,  1], 
           [-0.57114584,  1.15404176, -0.87468506,  1], 
           [-0.94714779,  1.13832305, -0.58694270,  1], 
           [-0.40102286,  1.46159431, -0.69792237,  1], 
           [-0.26437944,  1.34796154, -0.73774277,  1], 
           [-0.27293769,  1.59309487, -1.04274151,  1], 
           [-0.30283821, -0.73600306,  1.44039980,  2], 
           [-0.58290259, -0.25539981,  1.50781368,  2], 
           [-1.08018241, -0.25738737,  1.09149484,  2], 
           [-0.76039084, -0.97361097,  1.28860632,  2], 
           [-1.04975329, -0.77085457,  1.78097885,  2], 
           [-0.40436016, -0.52396243,  1.51026685,  2], 
           [-0.83580163, -0.84298958,  1.05583722,  2], 
           [-0.84797325, -0.47850486,  1.55482311,  2], 
           [-0.27698582, -0.67465935,  1.74025219,  2], 
           [-0.94195261, -0.72946186,  1.79217650,  2], 
           [-0.48707297, -0.86887812,  1.20341262,  2], 
           [-0.82242448, -0.75027166,  1.49045897,  2]]
dataset = np.array(dataset)

# Separando entradas e rótulos
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

# One-Hot Encoding (representar variáveis categóricas em um formato numérico binário)
num_classes = 3
y_encoded = np.zeros((len(y), num_classes))
for i, label in enumerate(y):
    y_encoded[i, label] = 1

#inicializando pesos
def layer_w(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count+1))
    for i in range(neuron_count):
        for j in range(1, (input_count+1)):
            weights[i][j] = np.random.uniform(-0.1, 0.1)
    return weights

#forward propagation
def forward_pass(x):
    global hidden_layer_y
    global output_layer_y

    # Activation function for hidden layer
    for i, w in enumerate(hidden_layer_w):
        z = np.dot(w, x)
        hidden_layer_y[i] = np.tanh(z)
    hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_y))

    # Activation function for output layer
    for i, w in enumerate(output_layer_w):
        z = np.dot(w, hidden_output_array)
        output_layer_y[i] = 1.0 / (1.0 + np.exp(-z))

#backpropagation
def backward_pass(y_truth):
    global hidden_layer_error
    global output_layer_error

    # Backpropagate error for each output neuron
    # and create array of all output neuron errors.
    for i, z in enumerate(output_layer_y):
        error_prime = -(y_truth[i] - z) # Loss derivative
        derivative = z * (1.0 - z) # Logistic derivative
        output_layer_error[i] = error_prime * derivative
    
    # Create array weights connecting the output of
    # hidden neuron i to neurons in the output layer.
    for i, z in enumerate(hidden_layer_y):
        error_weights = []
        for w in output_layer_w:
            error_weights.append(w[i+1])
        error_weight_array = np.array(error_weights)

        # Backpropagate error for hidden neuron.
        derivative = 1.0 - z**2 # tanh derivative
        weighted_error = np.dot(error_weight_array, output_layer_error)
        hidden_layer_error[i] = weighted_error * derivative

#atualizar os pesos da rede neural durante o treinamento
def adjust_weights(x):
    global output_layer_w
    global hidden_layer_w
    for i, error in enumerate(hidden_layer_error):
        hidden_layer_w[i] -= (x * LEARNING_RATE * error) # Update all weights
    hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_y))
    for i, error in enumerate(output_layer_error):
        output_layer_w[i] -= (hidden_output_array * LEARNING_RATE * error) # Update all weights

# Read train and test examples
index_list = list(range(len(X)))

# Declare matrices and vectors representing the neurons.
hidden_layer_w = layer_w(4, 3)  # 4 neurônios na camada oculta, 3 entradas
hidden_layer_y = np.zeros(4)  # Saídas da camada oculta (4 neurônios)
hidden_layer_error = np.zeros(4)  # Erros da camada oculta (4 neurônios)

output_layer_w = layer_w(3, 4)  # 3 neurônios na camada de saída, 4 entradas da camada oculta
output_layer_y = np.zeros(3)  # Saídas da camada de saída (3 neurônios)
output_layer_error = np.zeros(3)  # Erros da camada de saída (3 neurônios)

# Network training loop
for i in range(EPOCHS): # Train EPOCHS iterations
    np.random.shuffle(index_list) # Randomize order
    correct_training_results = 0
    for j in index_list: # Train on all examples
        x = np.concatenate((np.array([1.0]), X[j]))
        forward_pass(x)
        if output_layer_y.argmax() == y_encoded[j].argmax():
            correct_training_results += 1
        backward_pass(y_encoded[j])
        adjust_weights(x)


#receber a nova instancia
x1,x2,x3 = map(float,input().split())
new_instance = np.array([x1,x2,x3])
new_instance_with_bias = np.concatenate((np.array([1.0]), new_instance))

# Passa a nova instância pela rede
forward_pass(new_instance_with_bias)
if output_layer_y.argmax() == y_encoded[j].argmax():
        correct_training_results += 1
backward_pass(y_encoded[j])
adjust_weights(new_instance_with_bias)

predicted_class = np.argmax(output_layer_y)
print(predicted_class)