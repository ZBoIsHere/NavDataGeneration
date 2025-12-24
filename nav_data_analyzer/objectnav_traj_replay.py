import argparse
import re
from turtle import st
import habitat
import os
import attr
import json
import gzip

from habitat.utils.visualizations.utils import observations_to_image, images_to_video, append_text_to_image

import uuid

# UUID 生成逻辑
def generate_task_uuid(scene_type, scene_name, split, start_position, start_rotation, object_category):
    namespace = uuid.NAMESPACE_DNS
    key = f"{scene_type}:{scene_name}:{split}:{start_position}:{start_rotation}:{object_category}"
    return str(uuid.uuid5(namespace, key))

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


def make_metrics(info, episode):
    ep_json = attr.asdict(episode)
    ep_json['metrics'] = {
        'success': int(info['success']),
        'spl': info['spl'],
        'distance_to_goal': info['distance_to_goal'],
        'traj_len': len(episode.reference_replay)
    }
    del ep_json['_shortest_path_cache']
    return ep_json


def check_traj_format(args, episode):
    invalid_format = []

    filename = episode.scene_id
    replay = episode.reference_replay
    episode_id = episode.episode_id

    if len(replay) < 2:
        print(
            f"Warning: {filename} - Episode {episode_id} has less than 2 actions in reference_replay.")
        invalid_format.append(
            f"Episode {episode_id} has less than 2 actions in reference_replay.")
        return invalid_format

    first_action = replay[0].get('action')
    if first_action != 'STOP':
        invalid_format.append(
            f"Episode {episode_id} first action is not STOP, it is {first_action}")
        print(
            f"Warning: {filename} - Episode {episode_id} first action is not STOP, it is {first_action}")
    
    last_action = replay[-1].get('action')
    if last_action != 'STOP':
        invalid_format.append(
            f"Episode {episode_id} last action is not STOP, it is {last_action}")
        print(
            f"Warning: {filename} - Episode {episode_id} last action is not STOP, it is {last_action}")

    if len(replay) >= 500:
        invalid_format.append(
            f"Episode {episode_id} exceed 500 steps")
        print(
            f"Warning: {filename} - Episode {episode_id} exceed 500 steps")

    #for step in replay[1:-1]:
    #    action = step.get('action')
    #    if action not in ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN']:
    #        invalid_format.append(
    #            f"Episode {episode.get('episode_id')} has invalid action {action}")
    #        print(
    #            f"Warning: {filename} - Episode {episode.get('episode_id')} has invalid action {action}")

    return invalid_format


def traj_replay(args, cfg, batch_size, batch_id):
    print(f'process batch id: {batch_id}')
    scene_name = args.scene_name
    print(f'process scene {scene_name}')
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    with habitat.Env(cfg) as env:
        dataset = load_dataset(cfg.DATASET.DATA_PATH)
        dataset['episodes'] = []
        distance_to = cfg.TASK.DISTANCE_TO_GOAL.DISTANCE_TO

        import time
        for ep_id in range(len(env.episodes)):
            start_time = time.time()
            if (ep_id % batch_size) != batch_id and batch_size > 1:
                env.reset()
                continue
            env.reset()

            if args.save_metric == 'False' and args.save_video == 'True' and (ep_id % args.save_video_interval != 0):
                print(f'skip episode {ep_id} due to save_video_interval')
                env.reset()
                continue

            invalid_format = check_traj_format(args, env.current_episode)
            if len(invalid_format) > 0:
                print(f'skip episode {ep_id} due to invalid format')
                continue
            episode = env.current_episode
            # print episode_id and batch_id
            print(f'processing episode id: {episode.episode_id} in batch id: {batch_id}')
            info = {}

            observation_list = []
            closest_goal_object_id = episode.info['closest_goal_object_id']
            closest_goal_object_position = []
            if args.save_video == 'True':
                for goal in episode.goals:
                    if goal.object_id == closest_goal_object_id:
                        closest_goal_object_position = goal.position
                        break
                closest_goal_object_position = ', '.join(
                    [f'{x:.2f}' for x in closest_goal_object_position])

            num = 0
            for data in episode.reference_replay[1:]:  # skip the first stop
                num = num + 1
                #print(f'episode {ep_id} step {num}/{len(episode.reference_replay)-1}')
                action = possible_actions.index(data["action"])
                action_name = env.task.get_action_name(
                    action
                )

                observations = env.step(action=action)

                info = env.get_metrics()
                if args.save_video == 'True':
                    frame = observations_to_image(
                        {"rgb": observations["rgb"]}, info)

                    frame = append_text_to_image(
                        frame, f'closest_object_name: {episode.object_category}_{episode.info["closest_goal_object_id"]}; object_center: [{closest_goal_object_position}]'
                    )

                    position = env.sim.get_agent_state(0).position
                    sim_agent_position_str = ', '.join(
                        [f'{x:.2f}' for x in position])

                    frame = append_text_to_image(
                        frame, f'action: {data["action"]}; position: [{sim_agent_position_str}]'
                    )

                    frame = append_text_to_image(
                        frame, f'success {info["success"]}; spl: {info["spl"]}; distance2goal({distance_to}): {info["distance_to_goal"]:.2f} '
                    )

                    observation_list.append(frame)
                
                # 如果action_name是STOP，则提前结束循环
                if action_name == "STOP":
                    break

            if args.save_video == 'True' and (ep_id % args.save_video_interval == 0):
                video_path = os.path.join(
                    args.video_path, scene_name, episode.object_category, get_success_str(info['success']))
                
                uid = generate_task_uuid("", "", "", episode.start_position, episode.start_rotation, episode.object_category)
                video_name = f'{scene_name}_{episode.object_category}_{get_success_str(info["success"])}_spl-{info["spl"]:.2f}_step-{len(episode.reference_replay)}_dist-{info["distance_to_goal"]:.3f}_id-{uid}'
                images_to_video(observation_list, video_path, video_name)
            if args.save_metric == 'True':
                metric_json = make_metrics(info, episode)
                dataset['episodes'].append(metric_json)
            end_time = time.time()
            print(f'episode {ep_id} processed in {end_time - start_time:.2f} seconds')

        return dataset

def gen_batch_size():
    # cpu cores / gpus num / 2
    return 6

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_path", type=str,
                        default="data/traj_datasets/objectnav/hm3d_v1_hd_l3mvn_refine_v100_30k/train/content2",)
    parser.add_argument("--scene_name", type=str,
                        default='1S7LAXRdDqK', help="like 1S7LAXRdDqK")
    parser.add_argument("--scene_type", type=str,
                        default='hm3d_v1', help="hm3d_v1 or hm3d_v2")
    parser.add_argument("--save_video", type=str,
                        default='True', help="whether to save video or not")
    parser.add_argument("--save_video_interval", type=int, default=1,
                        help="the step of the episodes, like 2 means every two episodes")
    parser.add_argument("--save_metric", type=str,
                        default='False', help="whether to save metric or not")
    # allow_sliding
    parser.add_argument("--allow_sliding", type=str,
                        default='False', help="whether to allow sliding or not")

    args = parser.parse_args()

    if args.save_video == 'True':
        #args.video_path = args.traj_path.replace('content', 'content_videos')
        args.video_path = f"{args.traj_path.rstrip('/')}_videos"
        os.makedirs(args.video_path, exist_ok=True)

    if args.save_metric == 'True':
        #args.metrics_path = args.traj_path.replace('content', 'content_metrics')
        args.metrics_path = f"{args.traj_path.rstrip('/')}_metrics"
        os.makedirs(args.metrics_path, exist_ok=True)

    cfg = habitat.get_config(
        f"NavTrajSampleGeneration/L3MVN/envs/habitat/configs/tasks/objectnav_{args.scene_type}.yaml")
    cfg.defrost()
    cfg.DATASET.DATA_PATH = os.path.join(args.traj_path, args.scene_name + '.json.gz')
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = 3000
    if args.allow_sliding == 'True':
        cfg.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = True
    cfg.freeze()

    batch_size = gen_batch_size()
    print(f'generated batch size: {batch_size}')

    from multiprocessing import Pool
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    with Pool(processes=batch_size) as pool:
        results = pool.starmap(traj_replay, [(args, cfg, batch_size, batch_id) for batch_id in range(batch_size)])
        if args.save_metric == 'True':
            combined_dataset = results[0]
            for dataset in results[1:]:
                combined_dataset['episodes'].extend(dataset['episodes'])
            json_path = os.path.join(args.metrics_path, f'{args.scene_name}.json')
            write_json(combined_dataset, json_path)
            write_gzip(json_path, json_path)

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    end = time.time()
    print(f'Total time: {end - start} seconds')
