# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import argparse
import os
import gym
import time
import json
import dmc2gym

from logger import Logger
from video import VideoRecorder

from agent.baseline_agent import BaselineAgent
from agent.bisim_agent import BisimAgent
from agent.deepmdp_agent import DeepMDPAgent
from agents.navigation.carla_env import CarlaEnv


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--resource_files', type=str)
    parser.add_argument('--eval_resource_files', type=str)
    parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
    parser.add_argument('--total_frames', default=1000, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    # train
    parser.add_argument('--agent', default='bisim', type=str, choices=['baseline', 'bisim', 'deepmdp'])
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
    parser.add_argument('--load_encoder', default=None, type=str)
    # eval
    parser.add_argument('--eval_freq', default=10, type=int)  # TODO: master had 10000
    parser.add_argument('--num_eval_episodes', default=20, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str, choices=['pixel', 'pixelCarla096', 'pixelCarla098', 'identity'])
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    parser.add_argument('--encoder_stride', default=1, type=int)
    parser.add_argument('--decoder_type', default='pixel', type=str, choices=['pixel', 'identity', 'contrastive', 'reward', 'inverse', 'reconstruction'])
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.01, type=float)
    parser.add_argument('--alpha_lr', default=1e-3, type=float)
    parser.add_argument('--alpha_beta', default=0.9, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--transition_model_type', default='', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--port', default=2000, type=int)
    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step, device=None, embed_viz_dir=None, do_carla_metrics=None):
    # carla metrics:
    reason_each_episode_ended = []
    distance_driven_each_episode = []
    crash_intensity = 0.
    steer = 0.
    brake = 0.
    count = 0

    # embedding visualization
    obses = []
    values = []
    embeddings = []

    for i in range(num_episodes):
        # carla metrics:
        dist_driven_this_episode = 0.

        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)

            if embed_viz_dir:
                obses.append(obs)
                with torch.no_grad():
                    values.append(min(agent.critic(torch.Tensor(obs).to(device).unsqueeze(0), torch.Tensor(action).to(device).unsqueeze(0))).item())
                    embeddings.append(agent.critic.encoder(torch.Tensor(obs).unsqueeze(0).to(device)).cpu().detach().numpy())

            obs, reward, done, info = env.step(action)

            # metrics:
            if do_carla_metrics:
                dist_driven_this_episode += info['distance']
                crash_intensity += info['crash_intensity']
                steer += abs(info['steer'])
                brake += info['brake']
                count += 1

            video.record(env)
            episode_reward += reward

        # metrics:
        if do_carla_metrics:
            reason_each_episode_ended.append(info['reason_episode_ended'])
            distance_driven_each_episode.append(dist_driven_this_episode)

        video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)

    if embed_viz_dir:
        dataset = {'obs': obses, 'values': values, 'embeddings': embeddings}
        torch.save(dataset, os.path.join(embed_viz_dir, 'train_dataset_{}.pt'.format(step)))

    L.dump(step)

    if do_carla_metrics:
        print('METRICS--------------------------')
        print("reason_each_episode_ended: {}".format(reason_each_episode_ended))
        print("distance_driven_each_episode: {}".format(distance_driven_each_episode))
        print('crash_intensity: {}'.format(crash_intensity / num_episodes))
        print('steer: {}'.format(steer / count))
        print('brake: {}'.format(brake / count))
        print('---------------------------------')




def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        resource_files=args.resource_files,
        img_source=args.img_source,
        total_frames=args.total_frames,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.image_size,
        width=args.image_size,
        frame_skip=args.action_repeat
    )
    env.seed(args.seed)

    eval_env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        resource_files=args.eval_resource_files,
        img_source=args.img_source,
        total_frames=args.total_frames,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.image_size,
        width=args.image_size,
        frame_skip=args.action_repeat
    )

    # stack several consecutive frames together
    if args.encoder_type.startswith('pixel'):
        env = utils.FrameStack(env, k=args.frame_stack)
        eval_env = utils.FrameStack(eval_env, k=args.frame_stack)

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    done = True
    for step in range(args.num_train_steps):
        if done:
            obs = env.reset()
            done = False

        # sample action for data collection
        action = env.action_space.sample()

        curr_reward = reward
        next_obs, reward, done, _ = env.step(action)

        obs = next_obs



if __name__ == '__main__':
    main()
