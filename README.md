# segment-anything-2 real-time
Run Segment Anything Model 2 on a **live video stream using a realsense camera**


## Getting Started

### Installation

```bash
pip install -e .
```
### Download Checkpoint

Then, we need to download a model checkpoint.

```bash
cd checkpoints
./download_ckpts.sh
```

Then SAM-2-online can be used in a few lines as follows for image and video and **camera** prediction.

### Camera prediction

run ./demo/demo_realsense.py

## References:

- SAM2 Repository: https://github.com/facebookresearch/segment-anything-2
