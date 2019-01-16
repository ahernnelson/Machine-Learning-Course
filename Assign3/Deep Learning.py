# ### Forward propogation, simple example
import numpy as np
def relu(input):
     output = max(0, input)
     return(output)

# input_data = np.array([2,3])
# weights = {'node_0':np.array([1,1]),
#            'node_1':np.array([-1,1]),
#            'output':np.array([2,-1])}
#
# node_0_input = (input_data * weights['node_0']).sum()
# node_0_output = relu(node_0_input)
# node_1_input = (input_data * weights['node_1']).sum()
# node_1_output = relu(node_1_input)
# hidden_layer_outputs = np.array([node_0_output, node_1_output])
# print(hidden_layer_outputs)
#
#
# model_output = (hidden_layer_outputs * weights['output']).sum()
# print(model_output)
#
# ### generalzing to function
# input_data = None
# weights = None
# input_data = [np.array([3, 5]),np.array([ 1, -1]),np.array([0, 0]), np.array([8, 4])]
# weights = {'node_0':np.array([2, 4]),
#            'node_1':np.array([ 4, -5]),
#            'output':np.array([2, 7])}
# # Define predict_with_network()
# def predict_with_network(input_data_row, weights):
#     # Calculate node 0 value
#     node_0_input = (input_data_row * weights['node_0']).sum()
#     node_0_output = relu(node_0_input)
#
#     # Calculate node 1 value
#     node_1_input = (input_data_row * weights['node_1']).sum()
#     node_1_output = relu(node_1_input)
#
#     # Put node values into array: hidden_layer_outputs
#     hidden_layer_outputs = np.array([node_0_output, node_1_output])
#
#     # Calculate model output
#     input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
#     model_output = relu(input_to_final_layer)
#
#     # Return model output
#     return (model_output)
# # Create empty list to store prediction results
# results = []
# for input_data_row in input_data:
#     # Append prediction to results
#     results.append(predict_with_network(input_data_row, weights))
#
# # Print results
# print(results)
#
# #### Now with multi layer nn
# input_data = np.array([3,5])
# weights = {'node_0_0': np.array([2, 4]),
#  'node_0_1': np.array([ 4, -5]),
#  'node_1_0': np.array([-1,  2]),
#  'node_1_1': np.array([1, 2]),
#  'output': np.array([2, 7])}
# def predict_with_network(input_data):
#     # Calculate node 0 in the first hidden layer
#     node_0_0_input = (input_data * weights['node_0_0']).sum()
#     node_0_0_output = relu(node_0_0_input)
#
#     # Calculate node 1 in the first hidden layer
#     node_0_1_input = (input_data * weights['node_0_1']).sum()
#     node_0_1_output = relu(node_0_1_input)
#
#     # Put node values into array: hidden_0_outputs
#     hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
#
#     # Calculate node 0 in the second hidden layer
#     node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
#     node_1_0_output = relu(node_1_0_input)
#
#     # Calculate node 1 in the second hidden layer
#     node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
#     node_1_1_output = relu(node_1_1_input)
#
#     # Put node values into array: hidden_1_outputs
#     hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])
#
#     # Calculate model output: model_output
#     model_output = (hidden_1_outputs * weights['output']).sum()
#
#     # Return model_output
#     return (model_output)
#
#
# output = predict_with_network(input_data)
# print(output)
#
# ##### Adjusting weights intuitively
# # The data point you will make a prediction for
# input_data = np.array([0, 3])
#
# # Sample weights
# weights_0 = {'node_0': [2, 1],
#              'node_1': [1, 2],
#              'output': [1, 1]
#             }
#
# # The actual target value, used to calculate the error
# target_actual = 3
#
# # # Make prediction using original weights
# # model_output_0 = predict_with_network(input_data, weights_0)
# #
# # # Calculate error: error_0
# # error_0 = model_output_0 - target_actual
# #
# # # Create weights that cause the network to make perfect prediction (3): weights_1
# # weights_1 = {'node_0': [1, 1],
# #              'node_1': [1, 1],
# #              'output': [.5, .5]
# #             }
# #
# # # Make prediction using new weights: model_output_1
# # model_output_1 = predict_with_network(input_data, weights_1)
# #
# # # Calculate error: error_1
# # error_1 = model_output_1 - target_actual
# #
# # # Print error_0 and error_1
# # print(error_0)
# # print(error_1)
# #
# # ##### Scaline muliple data poinrts ############
# # input_data = [np.array([0, 3]), np.array([1, 2]),
# #               np.array([-1, -2]), np.array([4, 0])]
# # weights_0 = {'node_0': np.array([2, 1]),
# #              'node_1': np.array([1, 2]),
# #              'output': np.array([1, 1])}
# # weights_1 = {'node_0': np.np.array([2, 1]),
# #              'node_1': np.array([1. , 1.5]),
# #              'output': np.array([1. , 1.5])}
# # target_actuals = [1,3,5,7]
# from sklearn.metrics import mean_squared_error
# #
# # # Create model_output_0
# # model_output_0 = []
# # # Create model_output_0
# # model_output_1 = []
# #
# # # Loop over input_data
# # for row in input_data:
# #     # Append prediction to model_output_0
# #     model_output_0.append(predict_with_network(row, weights_0))
# #
# #     # Append prediction to model_output_1
# #     model_output_1.append(predict_with_network(row, weights_1))
# #
# # # Calculate the mean squared error for model_output_0: mse_0
# # mse_0 = mean_squared_error(target_actuals, model_output_0)
# #
# # # Calculate the mean squared error for model_output_1: mse_1
# # mse_1 = mean_squared_error(target_actuals, model_output_1)
# #
# # # Print mse_0 and mse_1
# # print("Mean squared error with weights_0: %f" % mse_0)
# # print("Mean squared error with weights_1: %f" % mse_1)
#
# ###### Calculate slopes and update weightss
# weights = np.array([1,2])
# input_dta = np.array([3,4])
# target = 6
# learning_rate = 0.01
# preds = (weights*input_data).sum()
# error = preds - target
# print(error)
#
# #### gradient
# gradient = 2 * input_data * error
# gradient
#
# weights_updated = weights - learning_rate * gradient
# preds_updated = (weights_updated * input_data).sum()
# error_updated = preds_updated - target
# print(error_updated)

#### another example
input_data = np.array([1,2,3])
weights = np.array([0,2,1])
target = 0
preds = (weights * input_data).sum()
error = preds - target
gradient = 2 * input_data * error
learning_rate = 0.01
weights_updated = weights - learning_rate * gradient
preds_updated = (weights_updated * input_data).sum()
error_updated = preds_updated - target
print(error)
print(error_updated)

##### generalizing process
from matplotlib import pyplot as plt
def get_slope(input_data, target, weights):
    preds = (weights * input_data).sum()
    error = preds - target
    return(2 * input_data * error)
def get_mse(input_data, target, weights):
    preds = (weights * input_data).sum()
    return(abs(preds - target))


n_updates = 20
mse_hist = []

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)

    # Update the weights: weights
    weights = weights - 0.01 * slope

    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)

    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()