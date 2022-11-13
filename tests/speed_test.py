import porlygon.environment as E
from porlygon.constants import IMG_SHAPE
from timeit import timeit 

env = E.DrawPolygonEnv(IMG_SHAPE, "data/jpg/", 100)
env.reset(seed = 114514)

action_space = env.action_space

def env_full_loop():
    done = False
    while not done:
        _, _, done, _ =  env.step(action_space.sample())
    env.reset()

if __name__ == "__main__":
    print(timeit("env_full_loop()", number = 1000, globals=globals()))
