import porlygon.environment as E
from porlygon.constants import IMG_SHAPE

if __name__ == "__main__":
    env = E.DrawPolygonEnv(IMG_SHAPE, "data/jpg/", 100, render_mode='human')
    env.reset(seed = 114514)
    
    action_space = env.action_space
    for i in range(100):
        done = False
        while not done:
            env.render()
            obs, reward, done, info =  env.step(action_space.sample())
        env.render()
        env.reset()
