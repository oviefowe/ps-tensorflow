import os
import cProfile
import tensorflow as tf

# Actual data
TRUE_W = 3.0
TRUE_B = 2.0
NUM_EXAMPLES = 10000

# Vector of random values
x = tf.random.normal(shape=[NUM_EXAMPLES])
noise = tf.random.normal(shape=[NUM_EXAMPLES])
y = x * TRUE_W + TRUE_B + noise

if __name__ == "__main__":
    print(y.numpy()[:10])