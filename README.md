## atari-agent

Tensorflow implementation of DQN

This implementation contains:

#### DQN
1. Deep Q-network and Q-learning
2. Experience replay memory
    - to reduce the correlations between consecutive updates
3. Network for Q-learning targets are fixed for intervals
    - to reduce the correlations between target and predicted Q-values
4. Double DQN
5. Dueling DQN


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

    $ python main.py

    or

    // Disable game window. Be able to improve training effect with GPU.
    $ python main.py --render=False

## Results

GPU: GTX 1060 3G

TBD

## References

- [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)
- [simple_dqn](https://github.com/tambetm/simple_dqn.git)
- [Code for Human-level control through deep reinforcement learning](https://sites.google.com/a/deepmind.com/dqn/)

## License

MIT License.
