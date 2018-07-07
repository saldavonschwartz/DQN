DQN: Deep Q-Network
===================

This project implements a variation on DQN (as described in the original `DeepMind paper <https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>`_) and learns several policies to solve
three reinforcement learning problems in OpenAI Gym: CartPole-v0, LunarLander-v2 and MountainCar-v0.

For a detailed writeup on the algorithm and project see `this post <https://0xfede.io/2018/05/17/dqn.html>`_.

The differences between this algorithm and DQN are:

1. no convolutional layers in the neural networks.
2. no preprocessing function (phi).
3. L2 squared loss by default

1 and 2 are not needed for the selected Gym environments since these already provide feature-based representations of states.
3 is just how I decided to train these agents. But the training algorithm allows replacing the L2 loss with any other loss, including Huber loss.

Training was done over combinations of epsilon annealing schedules and total training steps. For all environments,
three policies (models) are included: best performing, worst performing and one generated from a combination of hyperparameters
which proved successful in generating good policies for the three environments.

**Contents:**

- :code:`dqn.py`: implementation of DQN. Contains both training and testing functions.
- :code:`train.py`: training on all 3 environments at once.
- :code:`test.py`: testing on all 3 environments at once.
- :code:`plot.py`: plotting of training statistics.
- *CartPole-v0*: trained models along with training and testing stats for this environment.
- *LunarLander-v2*: trained models along with training and testing stats for this environment.
- *MountainCar-v0*: trained models along with training and testing stats for this environment.


Dependencies:
=============
* `OpenAI Gym <https://github.com/openai/gym>`_ (see installation note below)
* `Numpy <http://www.numpy.org>`_
* `NNKit <https://github.com/saldavonschwartz/nnkit>`_
* `matplotlib <www.apple.com>`_ (if you want to plot training stats).


Installation:
=============
After downloading or cloning the repo:

:code:`pip install -r requirements.txt`

**note:** in order to install Box2D (used by Gym for LunarLander), you might have:

1. download the Gym sources
2. compile and install Gym locally from sources: :code:`pip install -e <path to gym>`
3. install Box2D from Gym: :code:`pip install -e '<path to gym>/[box2d]'`


