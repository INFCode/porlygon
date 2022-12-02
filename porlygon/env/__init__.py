from gym.envs.registration import register

register(
    id='DrawPolygon-v0',
    entry_point='porlygon.env.polygon_env:DrawPolygonEnv',
)
