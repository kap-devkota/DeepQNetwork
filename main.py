import gym
from deep_q_net import DQN
from atari_wrappers import WarpFrame, FrameStack

EPISODES = 1000
MAX_FRAMES = 2000
EPSILON = .95
EPSILON_DECAY = .99
EPSILON_MIN = .05
BATCH_SIZE = 128
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
        2560,
        BATCH_SIZE)

    reward_episodes = []
    for i in range(EPISODES):
        state = env.reset()

        # Collection of datapoints
        for j in range(MAX_FRAMES):
            action = dqn.get_action(state)

            # Apply this action to the environment, get the next state and
            # reward, preprocess the next state
            next_state, reward, is_term, _ = env.step(action)

            dqn.store(state, action, next_state, reward, is_term)

            if is_term:
                break
            # Change to next state
            state = next_state

        # Training the model after data collection
        dqn.train()

        # Testing our model after training it
        state = env.reset()
        reward_per_epoch = 0
        for j in range(MAX_FRAMES):
            env.render()
            action = dqn.get_action(state, is_train=False)
            next_state, reward, is_term, _ = env.step(action)

            reward_per_epoch += (reward if reward >= 0 else reward * 10)

        print("------REWARD------\n=>" + str(reward_per_epoch))
        reward_episodes.append(reward_per_epoch)

    dqn.save()
    env.close()


if __name__ == '__main__':
    main()
