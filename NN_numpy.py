
# coding: utf-8

# In[19]:


import numpy as np 

beautiful  = 0.0
smart = 1.0
funny = 1.0

def old_activation_function(x):
    if x >= 0.5:
        return 1
    else:
        return 0 
    
def predict(vodka,rain,friend):
    #takes our input 
    inputs = np.array((beautiful, smart, funny))
    
    #BUILDING MODELS 
    #creating hidden neurons and specifying weights : neuron 1 and 2
    weights_input_to_hidden_1 = [0.25, 0.25, 0]
    weights_input_to_hidden_2 = [0.5, -0.4, 0.9]
    
    # creating matrix of 2 hidden neurons 
    weights_input_to_hidden = np.array((weights_input_to_hidden_1, weights_input_to_hidden_2))
    
    #last connection from hidden to ouput neuron 
    weights_hidden_to_output = np.array((-1, 1))
    
    #CALCULATION & TAKING ALL TOGETHER
    #to count what we will get from our hidden layer we need to multiply our input with weights
    hidden_input = np.dot(weights_input_to_hidden, inputs)
    print('hidden_inputs ' + str(hidden_input))
    
    # consist out of 2 elements, bc 2 hidden layer neurons 
    hidden_output = np.array([old_activation_function(x) for x in hidden_input])
    print('hidden_outputs ' + str(hidden_output))
    
    # last decision, 1 or 0 
    output = np.dot(weights_hidden_to_output, hidden_output)
    print('output ' + str(output))
    
    #pass our input through activation function 
    return old_activation_function(output) == 1

print('results: ' + str(predict(beautiful, smart, funny)))

