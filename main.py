from __future__ import absolute_import

import importlib
import datetime
import pdb
import random
import logging
import timeit
import time
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from agents.utils import variable_summaries
from agents.memory import Memory
from agents.history import History
from agents.environment import Environment

from inflection import underscore
from tensorflow.python import debug as tf_debug

import gym
from tqdm import tqdm
from gym import wrappers

flags = tf.app.flags
timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())

# DQN
# flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
# flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
# flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')
flags.DEFINE_string('agent_name', 'DoubleDuelingDQNAgent', 'THe name of agent to use')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
flags.DEFINE_string('log_dir', './log', 'Value of random seed')
flags.DEFINE_string('timestamp', timestamp, 'Timestamp')
FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

AVAILABLE_AGENT_LIST = [
        'RandomAgent',
        'DoubleDuelingDQNAgent',
        'A3cFfAgent',
        'A3cLstmAgent',
        ]


def main():
    """This is main function"""

    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    env = gym.make(FLAGS.env_name)

    with tf.Session() as sess:
        #  sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # tf.control_dependencies(None)
        agent_module = importlib.import_module(underscore('agents.' + FLAGS.agent_name))
        agent_klass = getattr(agent_module, FLAGS.agent_name)
        agent = agent_klass(env, sess, FLAGS)

        outdir = './results/%s/%s/%s/' % (
                FLAGS.env_name,
                str(agent_klass.__name__),
                timestamp
                #  hashlib.sha224(json.dumps(agent.config).encode('utf-8')).hexdigest()
                )

        env = wrappers.Monitor(env, directory=outdir, force=True)
        env.seed(0)

        total_steps = agent.config["total_steps"]
        episode_count = agent.config["num_episodes"]
        max_episode_length = agent.config["max_epLength"]

        ep_rewards = []
        actions = []

        ep_reward = 0.
        # e_tf = tf.placeholder(shape=[None], dtype=tf.float32)
        # loss_tf = tf.placeholder(shape=[None], dtype=tf.float32) 
        # r_tf = tf.placeholder(shape=[None], dtype=tf.float32)
        e_list = []
        loss_list = []
        # variable_summaries(e_tf, 'e')
        # variable_summaries(loss_tf, 'loss')
        # variable_summaries(r_tf, 'reward')

        total_reward = 0.
        reward = 0
        done = False
        episode_num = 0
        avg_reward = 0.
        avg_loss  = 0.
        avg_q = 0.
        avg_ep_reward, max_ep_reward, min_ep_reward = 0., 0., 0.

        agent.env = env
        memory = Memory()
        history = History()
        env = Environment(env)
        obs = env.reset()
        for _ in range(4):
            history.add(obs)
        agent.history = history
        agent.memory = memory

        #  for i in tqdm(range(episode_count)):
        for step in tqdm(range(total_steps), ncols=70, initial=0):

            if step == agent.config["pre_train_steps"]:
                num_game, agent.update_count, ep_reward = 0, 0, 0.
                total_reward, agent.total_loss, agent.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            action, obs, reward, done, _ = agent.act(env)
            total_loss, total_q, update_count, s1, loss, e = agent.learn(obs, reward, action, done)

            # e_list.append(e)
            # loss_list.append(loss)
            # ep_rewards.append(reward)

            if done:
                env.reset()
                episode_num += 1
                # episode_rewards_sum = np.sum(episode_rewards)
                # rewards.append(episode_rewards_sum)
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward

                # merged = tf.summary.merge_all()
                # summary = sess.run(merged, feed_dict={
                #     r_tf: rewards,
                #     e_tf: e_list,
                #     # loss_tf: loss_list,
                # })
                # agent.writer.add_summary(summary, episode_num)

            if step >= agent.config["pre_train_steps"]:
                if step % 2500 == 2500 - 1:
                    avg_reward = total_reward / 2500
                    avg_loss = total_loss / update_count
                    avg_q = total_q / update_count

                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    print('\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d, e: %.4f' \
                        % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, episode_num, e))

                    # if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                    #     self.step_assign_op.eval({self.step_input: self.step + 1})
                    #     self.save_model(self.step + 1)
                    #     max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

                    if step > 180:
                        agent.inject_summary({
                            'average.reward': avg_reward,
                            'average.loss': avg_loss,
                            'average.q': avg_q,
                            'episode.max reward': max_ep_reward,
                            'episode.min reward': min_ep_reward,
                            'episode.avg reward': avg_ep_reward,
                            'episode.num of game': episode_num,
                            'episode.rewards': ep_rewards,
                            'episode.actions': actions,
                            'training.learning_rate': agent.learning_rate_op.eval({agent.learning_rate_step: step}),
                            'e': e,
                        }, step)

                    episode_num = 0
                    total_reward = 0.
                    agent.total_loss = 0.
                    agent.total_q = 0.
                    agent.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []

            # if done:
            #     episode_num += 1
            #     episode_rewards_sum = np.sum(episode_rewards)
            #     rewards.append(episode_rewards_sum)
            #     episode_rewards = []
            #     env.reset()

                # merged = tf.summary.merge_all()
                # summary = sess.run(merged, feed_dict={
                #     r_tf: rewards,
                #     e_tf: e_list,
                #     # loss_tf: loss_list,
                # })
                # agent.writer.add_summary(summary, episode_num)

    env.close()

    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. \
            If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)



if __name__ == '__main__':
    main()
