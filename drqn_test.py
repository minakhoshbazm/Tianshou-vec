import argparse
import os
import pprint
from vec_env import vec_env
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.common import Net, Recurrent
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--stack-num', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--update-per-step', type=float, default=1 / 16)
    parser.add_argument('--step-per-collect', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=2)
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log/drqn')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    args = parser.parse_known_args()[0]
    return args


def test_drqn(args=get_args()):
    env = vec_env()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    net = Recurrent(args.layer_num, args.state_shape, args.action_shape,
                    args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq
    )
    # Let's watch its performance!
    env = DummyVectorEnv(
        [lambda: vec_env() for _ in range(1)]
    )
    
    policy.load_state_dict(torch.load('log/drqn/policy.pth'))
    policy.eval()
    policy.set_eps(args.eps_test)
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens, energies = result["rews"], result["lens"], result["energies"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}, energy: {energies.mean()}")


if __name__ == '__main__':
    test_drqn(args=get_args())