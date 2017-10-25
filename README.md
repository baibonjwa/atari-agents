## atari-agent

Tensorflow implementation of DQN, A3C

This implementation contains:

#### DQN 
1. Deep Q-network and Q-learning
2. Experience replay memory
    - to reduce the correlations between consecutive updates
3. Network for Q-learning targets are fixed for intervals
    - to reduce the correlations between target and predicted Q-values
4. Double DQN
5. Dueling DQN

#### A3C
1. LSTM
2. FC


## Requirements

- Python 2.7 or Python 3.3+
- [gym](https://github.com/openai/gym)
- [tqdm](https://github.com/tqdm/tqdm)
- [SciPy](http://www.scipy.org/install.html) or [OpenCV2](http://opencv.org/)
- [TensorFlow](https://github.com/tensorflow/tensorflow)


## Usage

First, install prerequisites with:

    $ pip install tqdm gym[all]

To train a model for Breakout:

    $ python main.py --env_name=Breakout-v0 --agent_name=TBD --is_train=True
    $ python main.py --env_name=Breakout-v0 --agent_name=TBD --is_train=True --display=True

To test and record the screen with gym:

    $ python main.py --is_train=False
    $ python main.py --is_train=False --display=True


## Results

Result of training for 24 hours using GTX 1060 ti.

## References

- [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)
- [simple_dqn](https://github.com/tambetm/simple_dqn.git)
- [Code for Human-level control through deep reinforcement learning](https://sites.google.com/a/deepmind.com/dqn/)

## License

MIT License.
