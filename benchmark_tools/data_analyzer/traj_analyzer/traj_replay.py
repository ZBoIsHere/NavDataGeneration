import argparse
import habitat
import os

from habitat.utils.visualizations.utils import observations_to_image, images_to_video, append_text_to_image


def make_videos(observations_list, scene_name, ep_id):
    prefix = scene_name + "_{}".format(ep_id)
    images_to_video(observations_list[0],
                    output_dir="/app/data/z00562901/NavDataGeneration/data/traj_datasets/objectnav/hm3d_v1_hd/episode_num_0-99_video", video_name=prefix)


def run_reference_replay(
    cfg, num_episodes=None, scene_name=None
):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    with habitat.Env(cfg) as env:
        total_success = 0
        spl = 0

        num_episodes = min(num_episodes, len(env.episodes))
        print("Replaying {}/{} episodes".format(num_episodes, len(env.episodes)))
        for episode_id in range(num_episodes):
            observation_list = []
            env.reset()

            # step_index = 1
            total_reward = 0.0
            episode = env.current_episode

            for data in env.current_episode.reference_replay:
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
                    frame, "Find and go to {}".format(episode.object_category))

                observation_list.append(frame)
                if action_name == "STOP":
                    break
            make_videos([observation_list], scene_name, episode.episode_id)
            print("Total reward for trajectory: {}".format(total_reward))

            if len(episode.reference_replay) <= 500:
                total_success += info["success"]
                spl += info["spl"]

        print("SPL: {}, {}, {}".format(spl/num_episodes, spl, num_episodes))
        print("Success: {}, {}, {}".format(total_success /
              num_episodes, total_success, num_episodes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_name", type=str, default='1S7LAXRdDqK'
    )
    parser.add_argument(
        "--traj_path", type=str, default="/app/data/z00562901/NavDataGeneration/data/traj_datasets/objectnav/hm3d_v1_hd/episode_num_0-99"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=3
    )

    config = habitat.get_config("/app/data/z00562901/NavDataGeneration/NavTrajSampleGeneration/L3MVN/envs/habitat/configs/tasks/objectnav_hm3d.yaml")
    args = parser.parse_args()
    cfg = config

    cfg.defrost()
    cfg.DATASET.DATA_PATH = os.path.join(
        args.traj_path, args.scene_name + '.json.gz')
    cfg.freeze()

    run_reference_replay(
        cfg,
        num_episodes=args.num_episodes,
        scene_name=args.scene_name
    )

if __name__ == "__main__":
    main()
