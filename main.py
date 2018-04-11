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
from agents.replay_memory import ReplayMemory
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
    # env = env.unwrapped
    # pdb.set_trace()

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
        e_list = []
        loss_list = []

        total_reward = 0.
        reward = 0
        done = False
        episode_num = 0
        episode_num_total = 0
        avg_reward = 0.
        avg_loss  = 0.
        avg_q = 0.
        avg_ep_reward, max_ep_reward, min_ep_reward = 0., 0., 0.

        agent.env = env
        memory = ReplayMemory()
        history = History()
        env = Environment(env)
        obs = env.reset()
        for _ in range(4):
            history.add(obs)
        agent.history = history
        agent.memory = memory
        merged = tf.summary.merge_all()

        #  for i in tqdm(range(episode_count)):
        for step_i in tqdm(range(total_steps), ncols=70, initial=0):

            if step_i == agent.config["pre_train_steps"]:
                episode_num, agent.update_count, ep_reward = 0, 0, 0.
                total_reward, agent.total_loss, agent.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            action, obs, reward, done, _ = agent.act(step_i, env)
            total_loss, total_q, update_count, s1, loss, e = agent.learn(step_i, obs, reward, action, done)

            if done:
                env.reset()
                episode_num += 1
                episode_num_total += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward

            # TODO: there is hard code
            if step_i >= agent.config["pre_train_steps"]:
                if step_i % 2500 == 2500 - 1:
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

                    episode_num = 0
                    total_reward = 0.
                    agent.total_loss = 0.
                    agent.total_q = 0.
                    agent.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []

                if done:
                    summary = sess.run(merged, feed_dict={
                        agent.summary_placeholders['ep.reward.avg']: avg_ep_reward,
                        agent.summary_placeholders['ep.reward.max']: max_ep_reward,
                        agent.summary_placeholders['ep.reward.min']: min_ep_reward,
                        agent.summary_placeholders['ep.num_of_game']: episode_num,
                        agent.summary_placeholders['avg.reward']: avg_reward,
                        agent.summary_placeholders['avg.loss']: avg_loss,
                        agent.summary_placeholders['avg.q']: avg_q,
                        agent.summary_placeholders['training.learning_rate']: agent.learning_rate_op.eval({agent.learning_rate_step: step_i}),
                        agent.summary_placeholders['e']: e,
                        agent.summary_placeholders['ep.rewards']: ep_rewards,
                        agent.summary_placeholders['ep.actions']: actions,
                    })
                    agent.writer.add_summary(summary, episode_num_total)

    env.close()

    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. \
            If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)



if __name__ == '__main__':
    main()
