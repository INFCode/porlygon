import time

import torch
import torch.nn as nn
import gym
import porlygon.env


def load_model(model_directoy: str) -> nn.Module:
    return torch.load(model_directoy)


if __name__ == "__main__":
    env = gym.make("DrawPolygon-v0", render_mode="human")
    obs, _ = env.reset(seed=42)

    model = load_model("path")

    term = False
    trunc = False
    while not term and not trunc:
        env.render()
        action = model(obs)
        obs, reward, term, trunc, info = env.step(action)
    env.render()
    time.sleep(5)
    env.reset()
    env.close()
