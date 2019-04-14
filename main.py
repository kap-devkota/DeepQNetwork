import gym
from deep_q_net import DQN
from collections import deque
from atari_wrappers import wrap_deepmind

PONG = 'Pong-v0'
NUM_FRAMES_TO_STACK = 4
EPISODES = 10000
MAX_FRAMES = 200
EPSILON = .95
EPSILON_DECAY = .999
EPSILON_MIN = .05
BATCH_SIZE = 128
NUM_SAMPLES_SCALE = 1
DEQUE_SIZE = 100000
GAMMA = .95

# If running on Gabe's laptop..
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


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

            if is_term or j == MAX_FRAMES - 1:
                running_reward = 0
                for k in reversed(range(len(temp))):
                    running_reward = temp[k][3] + GAMMA * running_reward
                    temp[k][3] = running_reward
                dqn.store(temp)
                break
            # Change to next state
            state = next_state
        dqn.train(NUM_SAMPLES_SCALE)
        if i % 100 == 0:
            dqn.save()


def get_env(game_name):
    env = gym.make(game_name)
    if game_name == PONG:
        env = wrap_deepmind(env, True, True, True, True)
    return env


if __name__ == '__main__':
    main()
