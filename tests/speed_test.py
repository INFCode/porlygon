from timeit import timeit
import gym
import porlygon.env

env = gym.make("DrawPolygon-v0")
env.reset(seed=42)

action_space = env.action_space


def env_full_loop():
    term = False
    trunc = False
    while not term and not trunc:
        _, _, term, trunc, _ = env.step(action_space.sample())
    env.reset()


if __name__ == "__main__":
    print(timeit("env_full_loop()", number=1000, globals=globals()))  # gives 223.23
