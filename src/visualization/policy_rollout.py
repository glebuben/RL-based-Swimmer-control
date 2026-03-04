import argparse
import gymnasium as gym
import torch
import os
import imageio


# ===== CAMERA CONFIGURATION =====
CAM_DISTANCE = 8.0
CAM_AZIMUTH = 90
CAM_ELEVATION = -20
CAM_LOOKAT = [1.0, 0.0, 0.0]
CAM_TRACK_BODY = -1
CAM_ID = -1
# ===============================


def fix_offscreen_camera(env):
    renderer = env.unwrapped.mujoco_renderer

    # Use free camera instead of model tracking camera
    renderer.camera_id = CAM_ID

    # Configure free camera
    renderer.default_cam_config = {
        "trackbodyid": CAM_TRACK_BODY,
        "distance": CAM_DISTANCE,
        "azimuth": CAM_AZIMUTH,
        "elevation": CAM_ELEVATION,
        "lookat": CAM_LOOKAT,
    }


def run_episode(env_name="Swimmer-v5", policy=None, max_steps=1000, gif_path=None):
    """Run one episode with optional neural network policy and save as GIF."""

    env = gym.make(env_name, render_mode="rgb_array")
    obs, info = env.reset()

    fix_offscreen_camera(env)

    # Ensure viewer is initialized
    frame = env.render()

    frames = []
    # print(dir(env.unwrapped.mujoco_renderer))
    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)

        if policy is None:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_tensor = policy(obs_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    env.close()

    if gif_path is not None and frames:
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Saved rollout GIF to: {gif_path}")


# ===============================
# CLI ENTRY POINT
# ===============================


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained policy in Gymnasium environment and save as GIF."
    )

    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",
        default=None,
        help="Path to policy checkpoint",
    )

    parser.add_argument(
        "--env",
        type=str,
        default="Swimmer-v5",
        help="Environment name",
    )

    args = parser.parse_args()

    # ---- load policy if provided ----
    policy = None
    gif_path = None

    if args.checkpoint is not None:
        from src.nn.nn_policy import ContinuousPolicy

        policy = ContinuousPolicy.load(args.checkpoint)
        policy.eval()
        print(f"Loaded policy from: {args.checkpoint}")

        # Save GIF near checkpoint
        ckpt_dir = os.path.dirname(args.checkpoint)
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        gif_path = os.path.join(ckpt_dir, f"{ckpt_name}_rollout.gif")
    # ---------------------------------

    run_episode(env_name=args.env, policy=policy, gif_path=gif_path)


if __name__ == "__main__":
    main()
