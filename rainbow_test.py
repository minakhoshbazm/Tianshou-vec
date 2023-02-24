import argparse
import os
import pickle
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import RainbowPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import NoisyLinear

from vec_env import vec_env

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--num-atoms', type=int, default=51)
    parser.add_argument('--v-min', type=float, default=-10.)
    parser.add_argument('--v-max', type=float, default=10.)
    parser.add_argument('--noisy-std', type=float, default=0.1)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.125)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
    )
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log/rainbow')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--prioritized-replay', action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--beta-final', type=float, default=1.)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument("--save-interval", type=int, default=4)
    args = parser.parse_known_args()[0]
    return args


def test_rainbow(args=get_args()):
    def noisy_linear(x, y):
        return NoisyLinear(x, y, args.noisy_std)
    env = vec_env()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        softmax=True,
        num_atoms=args.num_atoms,
        dueling_param=({
                           "linear_layer": noisy_linear
                       }, {
                           "linear_layer": noisy_linear
                       }),
    )
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = RainbowPolicy(
        net,
        optim,
        args.gamma,
        args.num_atoms,
        args.v_min,
        args.v_max,
        args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)
    # buffer
    env = DummyVectorEnv(
        [lambda: vec_env() for _ in range(1)]
    )

    policy.load_state_dict(torch.load('log/rainbow/policy.pth'))
    policy.eval()
    policy.set_eps(args.eps_test)
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens, energies = result["rews"], result["lens"], result["energies"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}, energy: {energies.mean()}")


if __name__ == '__main__':
    test_rainbow(args=get_args())