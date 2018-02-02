import os
import copy
import numpy as np
import pickle
import math
import IPython
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#constants/parameters for training
input_data_file = "AFE_core_5k_points.txt"
train_test_split_val = 65

learning_rate = 0.01
training_epochs = 300
batch_size = 1 
display_step = 1 

n_hidden_1 = 100 
n_hidden_2 = 100 
n_hidden_3 = 100 

def parse_training(file_name):
    if not os.path.isfile(file_name):
        print("File " + file_name + " doesnt exist")
        return
    f = open(file_name, "r")
    read_res = f.read()
    points_lst = read_res.split("\n")
    new_points_lst = []
    for p in points_lst[1:]:
        p = p.split("|")
        params = p[0].split(" ")
        params = [float(par) for par in params]
        results = p[1].split(" ")
        results = [float(res) for res in results]
        new_points_lst += [[params, results]]
    keys = points_lst[0].split("|")
    keys = [keys[0].split(" "), keys[1].split(" ")]


    n_param = len(keys[0])
    n_specs = len(keys[1])

    return keys, new_points_lst, n_param, n_specs




#Load and parse input data
def load_input(file_name,train_test_split):
    train_param = []
    test_param = []
    train_specs = []
    test_specs = []

    pickle_in = open(file_name,"rb")
    input_data_np = np.array(pickle.load(pickle_in))   
    input_data_np_norm = norm_data(input_data_np) 
    
    #arrange data as two separate arrays for input param and output specs
    #also split into training and test data sets
    train_split = int((float(train_test_split)/100)*(input_data_np_norm.shape[0]-1))

    #choose random indices for train and test assignment
    indices = random.sample(range(1,input_data_np_norm.shape[0]-1),input_data_np_norm.shape[0]-2)
    for index,i in enumerate(indices):
        if index < train_split:
            train_param.append(input_data_np_norm[i,0])
            train_specs.append(input_data_np_norm[i,1])
        else:
            test_param.append(input_data_np_norm[i,0])
            test_specs.append(input_data_np_norm[i,1])

    n_param = np.array(train_param).shape[1]
    n_specs = np.array(train_specs).shape[1]    
    
    return train_param,train_specs,test_param,test_specs,n_param,n_specs



#normalize data: y = (x - min)/(max - min), makes values between 0 and 1
def norm_data(input_data_np):
    min_vals_param = []
    max_vals_param = []
    min_vals_spec = []
    max_vals_spec = []
    
    reorg_params = np.zeros(((len(input_data_np[0,0]),input_data_np.shape[0])))
    reorg_specs = np.zeros(((len(input_data_np[0,1]),input_data_np.shape[0])))
    input_data_np_norm = copy.deepcopy(input_data_np) 
    for index,params_point in enumerate(input_data_np):
        if index != 0:
            for index2,param_val in enumerate(params_point[0]):
                reorg_params[index2,index] = param_val
            for index2,spec_val in enumerate(params_point[1]):
                reorg_specs[index2,index] = spec_val
    for param_vals in reorg_params: 
        max_vals_param.append(np.max(param_vals))
        min_vals_param.append(np.min(param_vals))
    for spec_vals in reorg_specs: 
        max_vals_spec.append(np.max(spec_vals))
        min_vals_spec.append(np.min(spec_vals))
    for index, params_point in enumerate(input_data_np):
        if index != 0:
            for index2, param_val in enumerate(params_point[0]):
                input_data_np_norm[index][0][index2] = float((param_val - min_vals_param[index2])/(max_vals_param[index2]-min_vals_param[index2]))
            for index2, spec_val in enumerate(params_point[1]):
                 input_data_np_norm[index][1][index2] = float((spec_val - min_vals_spec[index2])/(max_vals_spec[index2]-min_vals_spec[index2]))

    return input_data_np_norm

# Create model
def multilayer_perceptron(x, weights, biases):
    
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

#calculates MSE between predicted output of NN to the actual specs
def MSEerror(preds,train_specs):
    err_array = []
    for pred_epoch in preds:
        err = np.square(np.subtract(np.array(pred_epoch),np.array(train_specs)))
        err_array.append(np.sum(np.array(err)))
    min_err = np.min(np.array(err_array)) 
    return err_array,min_err 

#calculates average accuracy of predicted versus actual specs
def per_acc(preds,train_specs):
    per_acc_array = []
    train_specs = np.array(train_specs) + 10e-10
    for pred_epoch in preds:
        per_acc = np.divide(np.subtract(np.array(pred_epoch),np.array(train_specs)),np.array(train_specs))
        per_acc_array.append(1-abs(np.average(np.array(per_acc))))
    max_per_acc = np.max(np.array(per_acc_array))
    return per_acc_array,max_per_acc

def plot(error_array,per_acc_array):
    plt.subplot(2,1,1)
    plt.plot(range(0,training_epochs),error_array)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    
    plt.subplot(2,1,2)
    plt.plot(range(0,training_epochs),per_acc_array)
    plt.xlabel('Epochs')
    plt.ylabel('Percent Accuracy')
    plt.show()
    plt.close()

def main():
    # train_param, train_specs, test_param, test_specs,n_param,n_specs = load_input(input_data_file,train_test_split_val)
    keys, data, n_param, n_specs = parse_training(input_data_file)
    train_param = [d[0] for d in data]
    train_param = np.array(train_param).T.tolist()
    train_param = [[(d - min(p)) / (max(p)-min(p)) for d in p] for p in train_param]
    train_param = np.array(train_param).T.tolist()
    train_specs = [d[1] for d in data]
    train_specs = np.array(train_specs).T.tolist()
    train_specs = [[(d-min(p)) / (max(p)-min(p)) for d in p] for p in train_specs]
    train_specs = np.array(train_specs).T.tolist()

    train_specs, train_param = train_param, train_specs
    n_param, n_specs=n_specs, n_param
    print np.shape(train_param)
    print np.shape(train_specs)
    #Define TF inputs 
    x = tf.placeholder("float", [None, n_param])
    y = tf.placeholder("float", [None, n_specs])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_param, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_specs]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_specs]))
    }

    #construct model
    pred = multilayer_perceptron(x,weights,biases)

    #define loss and optimizer
    loss = tf.reduce_mean(tf.square(y-pred))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    #initialize training
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        costs = []
        preds = [] 
        #trainng cycle
        for i in range(training_epochs):
            _,cost,predic = sess.run([train,loss,pred],feed_dict={x:train_param,y:train_specs})
            costs.append(cost)
            preds.append((predic)) 
    
    err_array,min_err = MSEerror(preds,train_specs)
    avg_percent_diff,max_acc = per_acc(preds,train_specs)

    print('Min error: ' + str(min_err))
    print('Max accuracy: ' + str(max_acc) + '\n') 

    #plot MSE and percent accuracy 
    plot(err_array,avg_percent_diff)
    merged = tf.summary.merge_all()
    train_write = tf.summary.FileWriter('/tmp/krishnas/tf_test', sess.graph)
    test_write = tf.summary.FileWriter('/tmp/krishnas/tf_test')
    
if __name__=="__main__":
    main()
