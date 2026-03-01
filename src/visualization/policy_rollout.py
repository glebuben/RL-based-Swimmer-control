import argparse
import gymnasium as gym
import torch
import os
import imageio


# ===== CAMERA CONFIGURATION =====
CAM_DISTANCE = 8.0
CAM_AZIMUTH = 90
CAM_ELEVATION = -20
CAM_LOOKAT = [0.0, 0.0, 0.0]
CAM_TRACK_BODY = -1
# ===============================


def setup_fixed_camera(env):
    """Apply fixed camera settings to MuJoCo viewer."""
    viewer = env.unwrapped.mujoco_renderer.viewer
    cam = viewer.cam

    cam.trackbodyid = CAM_TRACK_BODY
    cam.distance = CAM_DISTANCE
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    cam.lookat[:] = CAM_LOOKAT


def run_episode(env_name="Swimmer-v5", policy=None, max_steps=1000, gif_path=None):
    """Run one episode with optional neural network policy and save as GIF."""

    env = gym.make(env_name, render_mode="rgb_array")
    obs, info = env.reset()

    # Ensure viewer is initialized
    frame = env.render()
    setup_fixed_camera(env)

    frames = [frame]

    for _ in range(max_steps - 1):
        # ---- choose action ----
        if policy is None:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_tensor = policy(obs_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()
        # -----------------------

        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)

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
