import gym
import numpy as np
from scipy import misc
from deep_q_net import DQN
from tensorflow import keras

# If running on Gabe's laptop..
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    env = gym.make('Pong-v0')
    dqn = DQN(env, 1000, 50000, .9, .25, 64, 10)
    dqn.train()
    # for i_episode in range(10):
    #     observation = env.reset()
    #     for t in range(100):
    #         env.render()
    #         action = env.action_space.sample()
    #         observation, reward, done, _ = env.step(action)
    #         if done:
    #             print("Episode finished after {} time steps".format(t + 1))
    #             break
    env.close()


def preprocess_pong_img(observation: np.ndarray) -> np.ndarray:
    """
    Takes an img observation and does some preprocessing. Crops the image,
    converts it to greyscale, and then scales it down.

    THIS ONLY WORKS FOR PONG.

    :param observation: The image observation, 210 X 160 X 3.
    :return: The greyscale 48 X 48 X 1 image.
    """
    cropped = observation[25:202, :, :]
    gray = np.mean(cropped, axis=2)
    scaled_down = misc.imresize(gray, (48, 48, 1))

    return scaled_down


def act(model: keras.models.Sequential,
        state: np.ndarray,
        env: gym.Env,
        epsilon: np.float):
    """
    Selects the action with the highest estimated reward with probability
    (1 - \epsilon) approximated by our model. Selects a random action from
    the state space with probability \epsilon.
    :param model:
    :param state: The current state. s_t
    :param env: The environment, used to grab a random action.
    :param epsilon: Probability a random action will be chosen. \epsilon
    :return:
    """
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    return np.argmax(model.predict(state))


if __name__ == '__main__':
    main()
