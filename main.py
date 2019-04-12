import gym
from deep_q_net import DQN
from collections import deque
from atari_wrappers import WarpFrame, FrameStack

EPISODES = 2000
MAX_FRAMES = 1000
EPSILON = .95
EPSILON_DECAY = .999
EPSILON_MIN = .05
BATCH_SIZE = 128
NUM_SAMPLES_SCALE = 10
DEQUE_SIZE = BATCH_SIZE * NUM_SAMPLES_SCALE * 2
GAMMA = .95


def main():
    env = gym.make('Pong-v0')
    env = WarpFrame(env)
    env = FrameStack(env, 4)

    action_list = range(env.action_space.n)
    dqn = DQN(
        action_list,
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
            env.render()
            action = dqn.get_action(state)

            # Apply this action to the environment, get the next state and
            # reward, preprocess the next state
            next_state, reward, is_term, _ = env.step(action)

            print("<TRAIN {} : {}>".format(i, action))

            temp.append([state, action, next_state, reward, is_term])

            if is_term or j == MAX_FRAMES - 1:
                running_reward = 0

                for k in reversed(range(len(temp))):
                    running_reward = temp[k][3] + GAMMA * running_reward
                    temp[k][3] = running_reward

                dqn.store(temp)
                state = env.reset()
                continue

            # Change to next state
            state = next_state

        # Training the model after data collection
        if i % 5 == 0:
            dqn.train(NUM_SAMPLES_SCALE)

            # Testing our model after training it
            state = env.reset()
            reward_per_epoch = 0
            for j in range(MAX_FRAMES):
                env.render()
                action = dqn.get_action(state, is_train=False)
                next_state, reward, is_term, _ = env.step(action)

                print("<TEST {} : {}>".format(i, action))

                if is_term:
                    break

                reward_per_epoch += (reward if reward >= 0 else reward * 10)

            print("------REWARD------\n=>" + str(reward_per_epoch))

    # env.close()


if __name__ == '__main__':
    main()
