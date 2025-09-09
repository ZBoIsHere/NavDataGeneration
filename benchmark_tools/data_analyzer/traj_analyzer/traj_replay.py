import argparse
from turtle import position

from matplotlib import category
import habitat
import os

from habitat.utils.visualizations.utils import observations_to_image, images_to_video, append_text_to_image

import json
import gzip

def precheck_dataset(dataset_path):
    ds = load_dataset(dataset_path)
    if len(ds['episodes']) == 0:
        return False
    return True

def get_success_str(info_success):
    if int(info_success) == 1:
        return 'success'
    else:
        return 'fail'

def write_json(data, path):
    with open(path, 'w') as file:
        file.write(json.dumps(data, indent=4))


def write_gzip(input_path, output_path):
    with open(input_path, "rb") as input_file:
        with gzip.open(output_path + ".gz", "wb") as output_file:
            output_file.writelines(input_file)


def load_dataset(path):
    with gzip.open(path, "rb") as file:
        data = json.loads(file.read(), encoding="utf-8")
    return data


import attr

def make_metrics(info, episode):
    ep_json = attr.asdict(episode)
    ep_json['metrics'] = {
        'success': info['success'],
        'spl': info['spl'],
        'distance_to_goal': info['distance_to_goal']
    }
    del ep_json['_shortest_path_cache']
    return ep_json


def run_reference_replay(
    args, cfg, scene_name=None
):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    with habitat.Env(cfg) as env:
        dataset = load_dataset(cfg.DATASET.DATA_PATH)
        dataset['episodes'] = []

        distance_to = cfg.TASK.DISTANCE_TO_GOAL.DISTANCE_TO

        #if args.episode_start > len(env.episodes):
        #    print(f'episode_start {args.episode_start} is larger than the number of episodes {len(env.episodes)}')
        #    return
        #if args.episode_end > len(env.episodes) or args.episode_end == -1:
        #    print(f'episode_end {args.episode_end} is larger than the number of episodes {len(env.episodes)}, set to {len(env.episodes)}')
        #    args.episode_end = len(env.episodes)

        #env.episodes = env.episodes[args.episode_start: args.episode_end: args.episode_step]
        print(f'len of selected episodes: {len(env.episodes)}')

        for ep_id in range(len(env.episodes)):
            if ep_id % args.episode_step != 0:
                print(f'skip episode {ep_id}')
                continue

            env.reset()
            observation_list = []

            episode = env.current_episode
           
            # 如果第一个action是stop，就把第一个stop去掉
            if episode.reference_replay[0]["action"] == "STOP" and len(episode.reference_replay) > 1:
                episode.reference_replay = episode.reference_replay[1:]

            closest_goal_object_id = episode.info['closest_goal_object_id']
            closest_goal_object_position = []
            for goal in episode.goals:
                if goal.object_id == closest_goal_object_id:
                    closest_goal_object_position = goal.position
                    break

            closest_goal_object_position = ', '.join([f'{x:.2f}' for x in closest_goal_object_position])

            info = {}
            for data in episode.reference_replay:
                print(f'data action is {data["action"]}')
                action = possible_actions.index(data["action"])
                action_name = env.task.get_action_name(
                    action
                )

                observations = env.step(action=action)

                info = env.get_metrics()
                frame = observations_to_image(
                    {"rgb": observations["rgb"]}, info)

                frame = append_text_to_image(
                    frame, f'closest_object_name: {episode.object_category}_{episode.info["closest_goal_object_id"]}; object_center: [{closest_goal_object_position}]'
                )

                position = env.sim.get_agent_state(0).position
                sim_agent_position_str = ', '.join([f'{x:.2f}' for x in position])

                frame = append_text_to_image(
                    frame, f'action: {data["action"]}; position: [{sim_agent_position_str}]'
                )

                frame = append_text_to_image(
                    frame, f'success {info["success"]}; spl: {info["spl"]}; distance2goal({distance_to}): {info["distance_to_goal"]:.2f} '
                )

                observation_list.append(frame)
                if action_name == "STOP":
                    break
            
            video_path = os.path.join(args.video_path, scene_name, episode.object_category, get_success_str(info['success']))
            video_name = f'{scene_name}_{episode.object_category}_{get_success_str(info["success"])}_spl-{info["spl"]:.2f}_step-{len(episode.reference_replay)}_id-{ep_id}'
            images_to_video(observation_list, video_path, video_name)
            #metric_json = make_metrics(info, episode)

            #dataset['episodes'].append(metric_json)

        #json_path = f'{args.traj_path}_metrics/{scene_name}.json'
        #write_json(dataset, json_path)
        #write_gzip(json_path, json_path)

import sys
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_name", type=str, default='vLpv2VX547B', help="like 1S7LAXRdDqK"
    )
    #parser.add_argument(
    #    "--episode_start", type=int, default=0, help="the start index of the episodes"
    #)
    #parser.add_argument(
    #    "--episode_end", type=int, default=-1, help="the end index of the episodes"
    #)
    parser.add_argument(
        "--episode_step", type=int, default=100, help="the step of the episodes, like 2 means every two episodes"
    )
    parser.add_argument(
        "--scene_type", type=str, default='hm3d_v2', help="hm3d_v1 or hm3d_v2"
    )
    parser.add_argument(
        "--traj_path", type=str, default="data/task_datasets/objectnav/hm3d_v2/train/content"
    )

    args = parser.parse_args()
    # video_path
    args.video_path = args.traj_path.replace('content', 'content_videos') 
    # mkdir video_path
    os.makedirs(args.video_path, exist_ok=True)

    config = habitat.get_config(f"NavTrajSampleGeneration/L3MVN/envs/habitat/configs/tasks/objectnav_{args.scene_type}.yaml")
    cfg = config

    cfg.defrost()
    cfg.DATASET.DATA_PATH = os.path.join(args.traj_path, args.scene_name + '.json.gz')
    #cfg.MAX_EPISODE_STEPS = 1000
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = 3000
    cfg.freeze()

    if not precheck_dataset(cfg.DATASET.DATA_PATH):
        print(f'dataset {cfg.DATASET.DATA_PATH} has no episodes, skip')
        sys.exit(0)


    run_reference_replay(
        args,
        cfg,
        scene_name=args.scene_name
    )

if __name__ == "__main__":
    main()
