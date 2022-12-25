import argparse
import os
import pprint

import gym
from gym import spaces
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.policy import TD3Policy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

import porlygon.env


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="DrawPolygon-v0")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--step-per-epoch", type=int, default=20000)
    parser.add_argument("--step-per-collect", type=int, default=8)
    parser.add_argument("--update-per-step", type=float, default=0.125)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--training-num", type=int, default=8)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--rew-norm", action="store_true", default=False)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_known_args()[0]
    return args


def make_actor(state_shape, action_shape, hidden_sizes, max_action, lr, device):
    net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = Actor(net, action_shape, max_action=max_action, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)
    return actor, actor_optim


def make_critic(state_shape, action_shape, hidden_sizes, lr, device):
    net = Net(
        state_shape, action_shape, hidden_sizes=hidden_sizes, concat=True, device=device
    )
    critic = Critic(net, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=lr)
    return critic, critic_optim


def test_td3(args=get_args()):
    task = "DrawPolygon-v0"
    env = gym.make(task)
    # TD3 only work on continuous action space
    assert isinstance(env.action_space, spaces.Box)
    args.state_shape = env.observation_space.shape
    # State shape is required
    assert args.state_shape is not None
    args.action_shape = env.action_space.shape
    args.max_action = env.action_space.high[0]
    if args.reward_threshold is None:
        args.reward_threshold = env.spec.reward_threshold
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)]
    )
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    actor, actor_optim = make_actor(
        args.state_shape,
        args.action_shape,
        args.hidden_sizes,
        args.max_action,
        args.critic_lr,
        args.device,
    )

    critic1, critic1_optim = make_critic(
        args.state_shape,
        args.action_shape,
        args.hidden_sizes,
        args.critic_lr,
        args.device,
    )

    critic2, critic2_optim = make_critic(
        args.state_shape,
        args.action_shape,
        args.hidden_sizes,
        args.critic_lr,
        args.device,
    )

    policy = TD3Policy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs)
    # train_collector.collect(n_step=args.buffer_size)
    # log
    log_path = os.path.join(args.logdir, args.task, "td3")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    # Iterator trainer
    trainer = OffpolicyTrainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )
    for epoch, epoch_stat, info in trainer:
        print(f"Epoch: {epoch}")
        print(epoch_stat)
        print(info)

    assert stop_fn(info["best_reward"])

    if __name__ == "__main__":
        pprint.pprint(info)
        # Let's watch its performance!
        env = gym.make(args.task)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    test_td3()
