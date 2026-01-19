# ROS/Gazebo and TurtleBot3 notes (zero-shot transfer)

This folder is provided as a **review-friendly placeholder** for the sim-to-real part of the pipeline.

In the paper, the best PPO policy is deployed **without fine-tuning** on:
1) the Python simulator (this repo),
2) a ROS/Gazebo digital twin, and
3) a TurtleBot3 Burger in a physical arena.

## What you should add here
- Your ROS2 distro (e.g., Humble), TurtleBot3 packages, and Gazebo version.
- The arena/world description (same geometry as Python simulator).
- A ROS node that:
  - subscribes to `/scan` (LiDAR),
  - constructs the same 120-bin discretized observation,
  - runs the trained SB3 policy (or exported TorchScript),
  - publishes `/cmd_vel`.

## Suggested minimal files
- `ros_gazebo/launch/bringup.launch.py`
- `ros_gazebo/worlds/arena.world`
- `ros_gazebo/policy_node/` (ROS2 package)

If you still have your trained checkpoint and vecnorm/normalization artifacts, place them under `checkpoints/` and reference them in the ROS node.
