import gym
from deep_q_net import DQN
from atari_wrappers import WarpFrame, FrameStack

EPISODES = 1000
MAX_FRAMES = 200
EPSILON = .95
EPSILON_DECAY = .99
EPSILON_MIN = .05
BATCH_SIZE = 64
GAMMA = .95


def main():
    env = gym.make('Pong-v0')
    env = WarpFrame(env)
    env = FrameStack(env, 4)

    action_list = range(env.action_space.n)
    dqn = DQN(
        action_list, EPSILON, EPSILON_DECAY, EPSILON_MIN, GAMMA, BATCH_SIZE)

    for i in range(EPISODES):
        state = env.reset()
        for j in range(MAX_FRAMES):
            env.render()
            action = dqn.get_action(state)

            # Apply this action to the environment, get the next state and
            # reward, preprocess the next state
            next_state, reward, is_term, _ = env.step(action)

            dqn.store_instance(state, action, next_state, reward, is_term)

            if is_term:
                break
            # Change to next state
            state = next_state
        dqn.train()
    env.close()


if __name__ == '__main__':
    main()
