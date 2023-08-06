import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import random
from tensorflow import keras


os.environ["CUDA_VISIBLE_DEVICES"]='0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


model = keras.models.load_model("./tmp.h5")
class DeltaDebugging4NN:
    def __init__(self, model, x_test, y_test):
        self.initial_model = model
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.class_nums = class_nums

    
    def acc(self):
        _loss, _acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return _acc

    
    def initial_acc(self):
        _loss, _acc = self.initial_model.evaluate(self.x_test, self.y_test, verbose=0)
        return _acc

    
    def zero_neurons(self, layer_indexes, neuron_indices):
        model = tf.keras.models.clone_model(self.initial_model)
        model.set_weights(self.initial_model.get_weights())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        fixed_value = 0.0 # making zero outputs
        
        for i in range(len(layer_indexes)): 
            layer_index = layer_indexes[i]
            neuron_index = neuron_indices[i]
            weights, biases = model.layers[layer_index].get_weights()
            
            if isinstance(model.layers[layer_index], tf.keras.layers.Conv2D):
                weights[:, :, :, neuron_index] = 0.0  
                biases[neuron_index] = fixed_value  
            else:
                weights[:, neuron_index] = 0.0  
                biases[neuron_index] = fixed_value  
            
            model.layers[layer_index].set_weights([weights, biases])
        self.model = model
        
        
    def isolate_neurons(self, layer_indexes, neuron_indices, internals_avg):
        model = tf.keras.models.clone_model(self.initial_model)
        model.set_weights(self.initial_model.get_weights())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        for i in range(len(layer_indexes)): 
            layer_index = layer_indexes[i]
            neuron_index = neuron_indices[i]
            fixed_value = internals_avg[layer_index][neuron_index]
            weights, biases = model.layers[layer_index].get_weights()
            
            if isinstance(model.layers[layer_index], tf.keras.layers.Conv2D):
                weights[:, :, :, neuron_index] = 0.0  
                biases[neuron_index] = fixed_value
            else:
                weights[:, neuron_index] = 0.0  
                biases[neuron_index] = fixed_value 
                
            model.layers[layer_index].set_weights([weights, biases])
        self.model = model
        
        
#     def neuron_estimition(self, layer_indexes, neuron_indices):    
#         import time 

#         start = time.time()
#         NS = []

#         for layer_index in layer_indexes:
#             print(layer_index)
#             l_class, l_overall = [], []
#             num_neurons = model.layers[layer_index].weights[0].numpy().shape[-1]
#             for idx in range(num_neurons):
#                 freeze_model.isolate_neurons(layer_indexes=[layer_index], neuron_indices=[idx])
#                 l_class.append(1 - freeze_model.ASR((1-mask)*x_train[:100] + trigger*mask, keras.utils.to_categorical([3]*100, 10)))
#             NR_backdoor.append(l_class)
#             print(l_class)
    
    
    def topk(self, NS, K):
        flattened_NS = np.concatenate(NS)
        topk_indexes = np.argsort(-flattened_NS)[:K]

        l_indexes = []
        n_indexes = []
        for i in layer_indexes:
            num_neurons = model.layers[i].weights[0].numpy().shape[-1]
            l_indexes.extend([i] * num_neurons)
            n_indexes.extend(range(0, num_neurons))

        NS_top_k_layer_indexes = [l_indexes[index] for index in topk_indexes]
        NS_top_k_neuron_indexes = [n_indexes[index] for index in topk_indexes]

        print(NS_top_k_layer_indexes)
        print(NS_top_k_neuron_indexes)
        
        return NS_top_k_layer_indexes, NS_top_k_neuron_indexes
    
    
    def egreedy(self, NS, K, epsilon=0.1): 
        flattened_NS = np.concatenate(NS)
        topk_indexes = np.argsort(-flattened_NR)[:A]
        topm_indexes = np.argsort(-flattened_NR)[A:B]
        rands = np.random.random(K)
        num_utilize = np.sum(rands > epsilon)
        num_explore = np.sum(rands <= epsilon)
        utilize_indexes = random.sample(list(topk_indexes), num_utilize)
        explore_indexes = random.sample(list(topm_indexes), num_explore)
        greedy = utilize_indexes + explore_indexes
        a = [l_indexes[index] for index in greedy]
        b = [n_indexes[index] for index in greedy]
        # freeze_model.isolate_neurons(a,b)
        # print(time.time()-start)
        
        return a, b
        
