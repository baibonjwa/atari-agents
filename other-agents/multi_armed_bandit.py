import tensorflow as tf
import numpy as np

bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)

def pullBandit(bandit):
    # Get a random number.
    result = np.random.randn(1)
    if result > bandit:
        return 1
    return -1

# agent
tf.reset_default_graph()
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights, 0)

reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
loss = -(tf.log(responsible_weight) * reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

# train
total_episodes = 10000
total_reward = np.zeros(num_bandits)
e = 0.1
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(total_episodes):
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)
        reward = pullBandit(bandits[action])

        _, resp, ww = sess.run(
            [update, responsible_weight, weights],
            feed_dict={reward_holder:[reward], action_holder: [action]})

        total_reward[action] += reward

        if i % 50 == 0:
            print("Running reward for the " + str(num_bandits) + " bandits " + str(total_reward))

print("The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising....")
if np.argmax(ww) == np.argmax(-1 * np.array(bandits)):
    print("...and it was right!")
else:
    print("...and it was wrong!")
