
""" Lean pad : taking Nielsen book examples and map them as closely as possible onto Tensorflow """

# pylint: disable=I0011,C0103

# Chapter 1 - complete
# Chapter 3 - incomplete / TBD : regularizations
#                                + SGD variations + alternate artificial neuron models
# Chapter 6 - TBD : conv layer + softmax layer

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

inputNeuronCount = 784
hiddenNeuronCount = 30
outputNeuronCount = 10
numberOfEpoch = 30
miniBatchSize = 10

gradient_type = "automatic"
#gradient_type = "manual"

activation_type = "sigmoid"

cost_type = "quadratic"
#cost_type = "cross_entropy"


if cost_type == "quadratic":
    # matches Nielsen hyper-parameter for quadratic cost function tests
    learningRate = 3.0

if cost_type == "cross_entropy":
    # matches Nielsen hyper-parameter for quadratic cost function tests
    learningRate = 0.5


# mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Nielsen Examples

a_0 = tf.placeholder(tf.float32, [None, inputNeuronCount])
w_1 = tf.Variable(tf.truncated_normal([inputNeuronCount, hiddenNeuronCount]))
b_1 = tf.Variable(tf.truncated_normal([1, hiddenNeuronCount]))
w_2 = tf.Variable(tf.truncated_normal([hiddenNeuronCount, outputNeuronCount]))
b_2 = tf.Variable(tf.truncated_normal([1, outputNeuronCount]))
y = tf.placeholder(tf.float32, [None, outputNeuronCount])

def sigmoid(x):
    """ the sigmoid function  : σ(x) = 1 / (1 + e−x) """
    return tf.sigmoid(x)

def sigmoidprime(x):
    """ derivative of the sigmoid function : σ′(z) = σ(z)(1−σ(z)) """
    return tf.mul(sigmoid(x), tf.sub(tf.constant(1.0), sigmoid(x)))

def activation_fn(x):
    """ activation function """
    if activation_type == "sigmoid":
        return sigmoid(x)
    return None

def activation_fn_prime(x):
    """ activation function prime """
    if activation_type == "sigmoid":
        return sigmoidprime(x)
    return None


# Forward propagation

# z1 = (a0 ⋅ w1) + b1
z_1 = tf.add(tf.matmul(a_0, w_1), b_1)

# a1 = σ(z1)
a_1 = activation_fn(z_1)

# z2 = (a1 ⋅ w2) + b2
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)

# a2 = σ(z2)
a_2 = activation_fn(z_2)


def log(x):
    """ the non-nan log function """
    return tf.log(tf.clip_by_value(x, 1e-10, 1.0))


def cost_fn(activation_result, desired_output):
    """ the cost function """

    if cost_type == "quadratic":
        return tf.div(tf.square(tf.sub(activation_result, desired_output)), tf.constant(2.0))

    if cost_type == "cross_entropy":
        arg1 = tf.mul(desired_output, log(activation_result))
        arg2 = tf.mul(tf.sub(tf.constant(1.0), desired_output),
                      log(tf.sub(tf.constant(1.0), activation_result)))
        return tf.neg(tf.add(arg1, arg2))

    return None




def cost_delta_last_layer_fn(activation_input, activation_result, desired_output):

    """ the cost delta function """

    if cost_type == "quadratic":
        return tf.mul(tf.sub(activation_result, desired_output),
                      activation_fn_prime(activation_input))

    if cost_type == "cross_entropy":
        return tf.sub(activation_result, desired_output)

    return None



# Use Tensorflow GradientDescentOptimizer to compute
# the gradient and perform the update

if gradient_type == "automatic":

    #reducedCost = tf.div(tf.reduce_sum(cost_fn1(a_2, y), reduction_indices=[0]),
    #                     tf.constant(1.0 * miniBatchSize))

    reducedCost = tf.reduce_mean(cost_fn(a_2, y), reduction_indices=[0])

    train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(reducedCost)


# Manually compute the gradient and perform the update
# (use Tensorflow instead of numpy, but it is equivalent)

if gradient_type == "manual":

    # Backward propagation equations

    # ∇z2 = ∇a ⋅ σ′(z2)
    d_z_2 = cost_delta_last_layer_fn(z_2, a_2, y)

    # ∇b2 = ∇z2
    d_b_2 = d_z_2

    # ∇w2 = aT1 ⋅ ∇z2
    d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

    # ∇a1 = ∇z2 ⋅ wT2
    d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))

    # ∇z1 = ∇a1 ⋅ σ′(z1)
    d_z_1 = tf.mul(d_a_1, activation_fn_prime(z_1))

    # ∇b1 = ∇z1
    d_b_1 = d_z_1

    # ∇w1 = aT0 ⋅ ∇z1
    d_w_1 = tf.matmul(tf.transpose(a_0), d_z_1)

    # updating the network

    # set the learning rate
    eta = tf.constant(learningRate / miniBatchSize)

    # the various operations conducted for each training run
    train_step = [
        # w1 ← w1 − η ⋅ ∇w1
        tf.assign(w_1, tf.sub(w_1, tf.mul(eta, d_w_1))),
        # b1 ← b1 − η ⋅ ∇b1
        tf.assign(b_1, tf.sub(b_1, tf.mul(eta, tf.reduce_mean(d_b_1, reduction_indices=[0])))),
        # w2 ← w2 − η ⋅ ∇w2
        tf.assign(w_2, tf.sub(w_2, tf.mul(eta, d_w_2))),
        # b2 ← b2 − η ⋅ ∇b2
        tf.assign(b_2, tf.sub(b_2, tf.mul(eta, tf.reduce_mean(d_b_2, reduction_indices=[0]))))
    ]


# Let's figure out where we predicted the correct label.
# tf.argmax is an extremely useful function which gives you the index of the highest entry
# in a tensor along some axis.
# For example, tf.argmax(a_2,1) is the label our model thinks is most likely for each input,
# while tf.argmax(y,1) is the correct label.
# We can use tf.equal to check if our prediction matches the truth.
accy_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))

# That gives us a list of booleans. To determine what fraction are correct,
# we cast to floating point numbers and then take the sum. For example, [True, False, True, True]
# would become [1,0,1,1] which would become 3.
accy_res = tf.reduce_sum(tf.cast(accy_mat, tf.float32))


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

numberOfIteration = round(len(mnist.train.images) / miniBatchSize)

print("Start training network using: ")
print(" - gradient type : {0}".format(gradient_type))
print(" - learning rate : {0}".format(learningRate))
print(" - activation function : {0}".format(activation_type))
print(" - cost function : {0}".format(cost_type))
print(" - hidden neuron count : {0}".format(hiddenNeuronCount))

# Run the complete training set multiple times (our network evolves at each iteration)
for iEpoch in range(numberOfEpoch):

    # loop until we've exhausted the training data
    for i in range(numberOfIteration):
        # extract a random set of training data
        batch_xs, batch_ys = mnist.train.next_batch(miniBatchSize)
        # feed the minibatch to our network
        sess.run(train_step,
                 feed_dict={a_0: batch_xs,
                            y: batch_ys})

    # display the progress for each epoch
    numberOfProperlyIdentifiedImages = sess.run(accy_res,
                                                feed_dict={a_0: mnist.test.images,
                                                           y: mnist.test.labels})
    print("Epoch {0}: {1} / {2}".format(iEpoch,
                                        numberOfProperlyIdentifiedImages, len(mnist.test.images)))
