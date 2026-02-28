import argparse
import gymnasium as gym
import torch


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


def run_episode(env_name="Swimmer-v5", policy=None, max_steps=1000):
    """Run one episode with optional neural network policy."""

    env = gym.make(env_name, render_mode="human")

    obs, info = env.reset()
    setup_fixed_camera(env)

    for _ in range(max_steps):

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

        if terminated or truncated:
            break

    env.close()


# ===============================
# CLI ENTRY POINT
# ===============================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained policy in Gymnasium environment."
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

    if args.checkpoint is not None:
        # Replace with your actual policy class import
        from your_project.policy import Policy

        policy = Policy.load(args.checkpoint)
        policy.eval()
        print(f"Loaded policy from: {args.checkpoint}")
    # ---------------------------------

    run_episode(env_name=args.env, policy=policy)


if __name__ == "__main__":
    main()