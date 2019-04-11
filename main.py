import gym
import numpy as np
from deep_q_net import DQN
from atari_wrappers import WarpFrame, FrameStack

EPISODES = 400
MAX_FRAMES = 1000
EPSILON = .95
EPSILON_DECAY = .99
EPSILON_MIN = .05
BATCH_SIZE = 256
NUM_SAMPLES_SCALE = 10
DEQUE_SIZE = BATCH_SIZE * NUM_SAMPLES_SCALE * 2
GAMMA = .95


def main():
    env = gym.make('CartPole-v1')
    # env = WarpFrame(env)
    # env = FrameStack(env, 4)

    action_list = range(env.action_space.n)
    dqn = DQN(
        action_list,
        EPSILON,
        EPSILON_DECAY,
        EPSILON_MIN,
        GAMMA,
        DEQUE_SIZE,
        BATCH_SIZE)

    dqn.load()
    for i in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, (1, 4))

        # Collection of datapoints
        for j in range(MAX_FRAMES):
            if i % 100 == 0:
                env.render()
            action = dqn.get_action(state)

            # Apply this action to the environment, get the next state and
            # reward, preprocess the next state
            next_state, reward, is_term, _ = env.step(action)
            next_state = np.reshape(next_state, (1, 4))

            dqn.store(state, action, next_state, reward, is_term)

            if is_term:
                state = env.reset()
            # Change to next state
            state = next_state

        # Training the model after data collection
        dqn.train(NUM_SAMPLES_SCALE)
    # dqn.save()
    # # Testing our model after training it
    # state = env.reset()
    # reward_per_epoch = 0
    # for j in range(MAX_FRAMES):
    #     env.render()
    #     action = dqn.get_action(state, is_train=False)
    #     next_state, reward, is_term, _ = env.step(action)
    #
    #     reward_per_epoch += (reward if reward >= 0 else reward * 10)
    #
    # print("------REWARD------\n=>" + str(reward_per_epoch))

    # env.close()


if __name__ == '__main__':
    main()
