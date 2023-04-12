import gym
import porlygon.env

if __name__ == "__main__":
    env = gym.make("DrawPolygon-v0", render_mode="human")
    env.reset(seed=42)

    action_space = env.action_space
    for i in range(2):
        term = False
        trunc = False
        while not term and not trunc:
            env.render()
            obs, reward, term, trunc, info = env.step(action_space.sample())
        env.render()
        env.reset()
    env.close()
