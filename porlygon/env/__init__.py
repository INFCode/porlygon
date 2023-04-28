from gym.envs.registration import register
from porlygon.env.env import DrawPolygonEnv

register(
    id="DrawPolygon-v0",
    entry_point="porlygon.env:DrawPolygonEnv",
)
