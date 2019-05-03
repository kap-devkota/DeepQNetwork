import gym
from deep_q_net import DQN
from collections import deque
from atari_wrappers import wrap_deepmind

PONG = 'Pong-v0'
NUM_FRAMES_TO_STACK = 4
EPISODES = 100000
MAX_FRAMES = 6000
EPSILON = .95
EPSILON_DECAY = .9999
EPSILON_MIN = .05
BATCH_SIZE = 128
NUM_SAMPLES_SCALE = 1
DEQUE_SIZE = 100000
GAMMA = .95


def main():
    env = get_env(game_name=PONG)

    num_actions = env.action_space.n
    dqn = DQN(
        num_actions,
        EPSILON,
        EPSILON_DECAY,
        EPSILON_MIN,
        GAMMA,
        DEQUE_SIZE,
        BATCH_SIZE)
    for i in range(EPISODES):
        state = env.reset()

        # Collection of datapoints
        temp = deque(maxlen=MAX_FRAMES)
        for j in range(MAX_FRAMES):
            # Apply this action to the environment, get the next state and
            # reward, preprocess the next state
            action = dqn.get_action(state)
            next_state, reward, is_term, _ = env.step(action)
            temp.append([state, action, next_state, reward, is_term])
            if reward != 0:
                print(reward)

            if is_term or j == MAX_FRAMES - 1:
                running_reward = 0
                for k in reversed(range(len(temp))):
                    if temp[k][3] != 0:
                        running_reward = 0
                    running_reward = temp[k][3] + GAMMA * running_reward
                    temp[k][3] = running_reward
                dqn.store(temp)
                break
            # Change to next state
            state = next_state
        dqn.train(NUM_SAMPLES_SCALE)
        if i % 1000 == 0:
            print("Episode: {}".format(i))
            print("Exploration: {}".format(dqn.exploration))
            dqn.save(i)
    dqn.save("last")


def get_env(game_name):
    """
    Wraps the environment in a couple of decorators formulated by deep mind, and
    implemented by OpenAi that perform preprocessing.

    :param game_name: The game that will be played.
    :return: The wrapped environment.
    """

    env = gym.make(game_name)
    if game_name == PONG:
        env = wrap_deepmind(env, False, False, True, False)
    return env


if __name__ == '__main__':
    main()
