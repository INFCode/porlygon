import unittest
import numpy as np
import torch
from td3_modules import *

class Td3TestHelper:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    obs_shape = (3, 128, 128)
    obs_intermed = 64
    act_shape = (12,)
    act_intermed = 16
    batch_size = 16

    @staticmethod
    def make_obs_preprocess(device = None) -> ObsPreprocessNet:
        device = device or Td3TestHelper.device
        obs_pre = ObsPreprocessNet(
            single_obs_shape=Td3TestHelper.obs_shape, 
            intermed_rep_size=Td3TestHelper.obs_intermed, device=device
        ).to(device)
        return obs_pre

    @staticmethod
    def make_act_preprocess(device = None) -> ActPreprocessNet:
        device = device or Td3TestHelper.device
        act_pre = ActPreprocessNet(
            single_act_shape=Td3TestHelper.act_shape, intermed_rep_size=Td3TestHelper.act_intermed, device=device
        ).to(device)
        return act_pre

    @staticmethod
    def make_actor(device = None) -> ActorNet:
        device = device or Td3TestHelper.device
        actor = ActorNet(
            Td3TestHelper.make_obs_preprocess(), 
            preprocess_net_output_dim=Td3TestHelper.obs_intermed, 
            action_shape=Td3TestHelper.act_shape
        ).to(device)
        return actor

    @staticmethod
    def make_critic(device = None) -> CriticNet:
        device = device or Td3TestHelper.device
        critic = CriticNet(
            act_preprocess_net=Td3TestHelper.make_act_preprocess(),
            act_preprocess_net_output_dim=Td3TestHelper.act_intermed,
            obs_preprocess_net=Td3TestHelper.make_obs_preprocess(),
            obs_preprocess_net_output_dim=Td3TestHelper.obs_intermed,
            device=device,
        ).to(device)
        return critic

    @staticmethod
    def random_obs() -> ObsT:
        obs: ObsT = {
            "reference": np.random.rand(Td3TestHelper.batch_size, 3, 128, 128),
            "canvas": np.random.rand(Td3TestHelper.batch_size, 3, 128, 128),
        }
        return obs

    @staticmethod
    def random_act() -> ActT:
        return np.random.rand(Td3TestHelper.batch_size, *Td3TestHelper.act_shape)

class TestObsPreprocessNet(unittest.TestCase):

    def test_output_size_match(self):
        obs_pre = Td3TestHelper.make_obs_preprocess()
        obs_logit, state = obs_pre(Td3TestHelper.random_obs())

        self.assertEqual(obs_logit.size(), (Td3TestHelper.batch_size, Td3TestHelper.obs_intermed))
        self.assertIs(state, None)

class TestActPreprocessNet(unittest.TestCase):

    def test_output_size_match(self):
        act_pre = Td3TestHelper.make_act_preprocess()
        act_logit, state = act_pre(Td3TestHelper.random_act())

        self.assertEqual(act_logit.size(), (Td3TestHelper.batch_size, Td3TestHelper.act_intermed))
        self.assertIs(state, None)

class TestActorNet(unittest.TestCase):

    def test_output_size_match(self):
        actor = Td3TestHelper.make_actor()
        action, state = actor(Td3TestHelper.random_obs())

        self.assertEqual(action.size(), (Td3TestHelper.batch_size,) + Td3TestHelper.act_shape)
        self.assertIs(state, None)

class TestCriticNet(unittest.TestCase):

    def test_output_size_match(self):
        critic = Td3TestHelper.make_critic()
        q_val = critic(Td3TestHelper.random_obs(), Td3TestHelper.random_act())

        self.assertEqual(q_val.size(), (Td3TestHelper.batch_size, 1))

if __name__ == "__main__":
    unittest.main()
