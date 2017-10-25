import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

#  from modelAny import *

#  from tensorflow.python.framework import dtypes
#  from tensorflow.python.framework import ops
#  from tensorflow.python.framework.ops import array_ops
#  from tensorflow.python.framework.ops import control_flow_ops
#  from tensorflow.python.framework.ops import embedding_ops
#  from tensorflow.python.framework.ops import math_ops
#  from tensorflow.python.framework.ops import nn_ops
#  from tensorflow.python.framework.ops import rnn
#  from tensorflow.python.framework.ops import rnn_cell
#  from tensorflow.python.framework.ops import variable_scope

import gym
env = gym.make('CartPole-v0')

H = 8
learning_rate = 1e-2
gamma = 0.99
decay_rate = 0.99
resume = False

model_bs = 3
real_bs = 3

D = 4

tf.reset_default_graph()
observations = tf.placeholder(tf.float32, [None, 4], name = "input_x")
W1 = tf.get_variable("W1", shape=[4, H],
        initializer = tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))

W2 = tf.get_variable("W2", shape=[H, 1],
        initializer = tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

mH = 256
input_data = tf.placeholder(tf.float32, [None, 5])
with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w", [mH, 50])
    softmax_b = tf.get_variable("softmax_b", [50])

previous_state = tf.placeholder(tf.float32, [None, 5], name="previous_state")
W1M = tf.get_variable("W1M", shape=[5, mH],
        initializer=tf.contrib.layers.xavier_initializer())
B1M = tf.Variable(tf.zeros([mH]), name="B1M")
layer1M = tf.nn.relu(tf.matmul(previous_state, W1M) + B1M)
W2M = tf.get_variable("W2M", shape=[mH, mH],
        initializer=tf.contrib.layers.xavier_initializer())
B2M = tf.Variable(tf.zeros([mH]), name="B2M")
layer2M = tf.nn.relu(tf.matmul(layer1M, W2M) + B2M)
wO = tf.get_variable("wO", shape=[mH, 4],
        initializer=tf.contrib.layer.xavier_initializer())
wR = tf.get_variable("wR", shape=[mH, 1],
        initializer=tf.contrib.layer.xavier_initializer())
wD = tf.get_variable("wD", shape=[mH, 1],
        initializer=tf.contrib.layer.xavier_initializer())

bO = tf.Variable(tf.zeros([4]), name="bO")
bR = tf.Variable(tf.zeros([1]), name="bR")
bD = tf.Variable(tf.ones([4]), name="bD")

predicted_observation = tf.matmul(layer2M, wO, name="predicted_observation") + bO
predicted_reward = tf.matmul(layer2M, wR, name="predicted_reward") + bR
predicted_done = tf.sigmoid(tf.matmul(layer2M, wD, name="predicted_done") + bD)

true_observation = tf.placeholder(tf.float32, [None, 4], name="true_observation")
true_reward = tf.placeholder(tf.float32, [None, 1], name="true_reward")
true_done = tf.placeholder(tf.float32, [None, 1], name="true_done")

predicted_state = tf.concat(1, [predicted_observation, predicted_reward, predicted_done])

observation_loss = tf.square(true_observation - predicted_observation)
reward_loss = tf.square(true_reward - predicted_reward)
done_loss = tf.mul(predicted_done, true_done) + tf.mul(1 - predicted_done, 1 - true_done)
done_loss = -tf.log(done_loss)

model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss)

modalAdam = tf.train.AdamOptimizer(learning_rate=learning_rate)
updateModel = modelAdam.minimize(model_loss)

def resetGradBuffer(gradBuffer):
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    return gradBuffer

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def stepModel(sess, xs, action):
    toFeed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1, 5])
    myPredict = sess.run([predicted_state], feed_dict={previous_state: toFeed})
    reward = myPredict[0][:, 4]
    observation = myPredict[0][:, 0:4]
    observation[:, 0] = np.clip(observation[:, 0], -2.4, 2.4)
    observation[:, 2] = np.clip(observation[:, 2], -0.4, 0.4)
    doneP = np.clip(myPredict[0][:, 5], 0, 1)
    if doneP > 0.1 or len(xs) >= 300:
        done = True
    else:
        done = False
    return observation, reward, done

xs, drs, ys, ds = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1
init = tf.initialize_all_variables()
batch_size = real_bs

drawFromModel = False
trainTheModel = True
trainThePolicy = False
switch_point = 1

with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()
    x = observation
    gradBuffer = sess.run(tvars)
    gradBuffer = resetGradBuffer(gradBuffer)

    while episode_number <= 5000:
        if (reward_sum/batch_size > 150 and drawFromModel == False) or rendering == True:
            env.render()
            rendering = True
        x = np.reshape(observation, [1, 4])
        tfprob = sess.run(probability, feed_dict = {observations: x})
        action = 1 if np.random.uniform() < tfprob else 0
        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)
        if drawFromModel == False:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done = stepModel(sess, xs, action)
        reward_sum += reward

        ds.append(done + 1)
        drs.append(reward)

        if done:
            if drawFromModel == False:
                real_episodes += 1
            episode_number += 1

            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            epd = np.vstack(ds)
            xs, drs, ys, ds = [], [], [], []

            if trainTheModel == True:
                actions = np.array([np.abs(y - 1) for y in epy][:-1])
                state_prevs = epx[:-1, :]
                state_prevs = np.hstack([state_prevs, actions])
                state_nexts = epx[1:, :]
                rewards = np.array(epr[1:, :])
                dones = np.array(epd[1:, :])
                state_nextsAll = np.hstack([state_nexts, rewards, dones])
                feed_dict = {previous_state: state_prevs, true_observation: state_nexts, true_done: dones, true_reward:rewards}
                loss, pState, _ = sess.run([model_loss, predicted_state, updateModel], feed_dict)
                if trainThePolicy == True:
                    discounted_epr = discount_rewards(epr).astype('float32')
                    discounted_epr -= np.mean(discounted_epr)
                    discounted_epr /= np.std(discounted_epr)
                    tGrad = sess.run(newGrads, feed_dict = {observations: epx, input_y: epy, advantages: disctouned_epr})
                    if np.sum(tGrad[0] == tGrad[0]) == 0:
                        break
                    for ix, grad in enumerate(tGrad):
                        gradBuffer[ix] += grad

                if switch_point + batch_size == episode_number:
                    switch_point = episode_number
                    if trainThePolicy == True:
                        sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad:gradBuffer[1]})
                        gradBuffer = resetGradBuffer(gradBuffer)

                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                    if drawFromModel == False:
                        print('Warld Perf: Episode %f. Reward %f. action: %f. mean reward %f.') & (real_episodes, reward_sum/real_bs, action, running_reward/real_bs)
                        if reward_sum/batch_size > 200:
                            break
                    reward_sum = 0

                    if episode_number > 100:
                        drawFromModel = not drawFromModel
                        trainTheModel = not trainTheModel
                        trainThePolicy = not trainThePolicy

                if drawFromModel == True:
                    observation = np.random.uniform(-0.1, 0.1, [4])
                    batch_size = model_bs
                else:
                    observation = env.reset()
                    batch_size = real_bs

print(real_episodes)
