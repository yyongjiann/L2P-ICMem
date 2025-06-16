# L2P-ICMem

This repository is heavily modified from the PyTorch implementation code for awesome continual learning method L2P,
Wang, Zifeng, et al. "Learning to prompt for continual learning." CVPR. 2022. We adapted the code to the ICMD dataset to enable memorability-aware training and replay.

The official Jax implementation is [here](https://github.com/google-research/l2p).

## Getting Started

Follow these steps to set up and run the project:

### 1. Set up Environment

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Requirements & Dataset

Install the required packages:

```bash
pip install -r requirements.txt
```

Download the ICMD dataset to the specified directory:

```bash
# Create the directory if it doesn't exist
mkdir -p baseline_1/local_datasets/

# Download your dataset here
# Example: wget [dataset_url] -P baseline_1/local_datasets/
# Or copy your dataset files to: baseline_1/local_datasets/
```

### 3. Run PBS Script

Execute the PBS script:

```bash
cd baseline_1
qsub l2p.pbs
```

## Acknowledgements

This project builds upon the work presented in the paper:

```
@inproceedings{wang2022learning,
  title={Learning to prompt for continual learning},
  author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={139--149},
  year={2022}
}

```
