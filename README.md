# Multi-finger Manipulation via Trajectory Optimization with Differentiable Rolling and Geometric Constraints

[📄 Paper](https://arxiv.org/pdf/2408.13229), [🌍 Weubsite](https://sites.google.com/umich.edu/multi-finger-rolling/home), [🎯 Benchmark](https://github.com/UM-ARM-Lab/MFR_benchmark)

MultiFingerTrajOpt is a trajectory optimization framework for multi-finger robotic manipulation tasks. It provides tools to optimize dexterous manipulation trajectories for robotic hands assuming a fixed contact mode. 

## Features
- Trajectory Optimization:
  - Supports multi-finger robotic hands for complex manipulation tasks.
- Task-Specific Implementations:
  - Cuboid alignment and turning.
  - Screwdriver turning.
  - Object inhand reorientation.
- Simulation Integration:
  - Compatible with NVIDIA Isaac Gym for high-performance physics simulation.

## Installation

### Prerequisites

Ensure the following dependencies are installed:
- Python 3.6+
- NVIDIA Isaac Gym Preview 4(Download here: https://developer.nvidia.com/isaac-gym)
- Pytorch Kinematics: https://github.com/UM-ARM-Lab/pytorch_kinematics. Please switch to the branch "chain_jacobian_at_links" and install from source: pip install -e .
- Pytorch Volumetric: https://github.com/UM-ARM-Lab/pytorch_volumetric. Please swtich to the branch "collision_cost" and install from source: pip install -e .
- torch_cg: https://github.com/sbarratt/torch_cg.
- MFR_benchmark: https://github.com/UM-ARM-Lab/MFR_benchmark
- CSVTO: https://github.com/UM-ARM-Lab/ccai. Please switch to the branch "multi_finger". This is the CSVTO solve (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10598358) we are using.
## Usage
### Example Scripts

Run the example scripts to test trajectory optimization for specific tasks. Switch to the optimization folder and then:
python eval.py --task='screwdriver_turning'


## Contact

For questions or issues, please contact the author at fanyangr@umich.edu.