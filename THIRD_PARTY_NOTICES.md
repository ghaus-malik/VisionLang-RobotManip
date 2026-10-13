# Third-party notices

All source code in this repository was written for this project. No code was copied from external
repositories. The project depends on the following third-party software and pretrained weights, which are
installed separately and are not redistributed here.

| Component | Use in this project | License |
|---|---|---|
| NVIDIA Isaac Sim 4.5 | physics simulation and rendering | NVIDIA Isaac Sim additional software and materials license |
| Isaac Lab 2.1 (`isaaclab`, `isaaclab_assets`) | scene, sensor, robot and IK-controller APIs; Franka Panda robot configuration and asset | BSD-3-Clause |
| PyTorch, torchvision | neural network training and inference | BSD-3-Clause |
| Hugging Face `transformers` | loading the CLIP model and tokenizer | Apache-2.0 |
| CLIP ViT-B/16 weights (`openai/clip-vit-base-patch16`) | frozen image and text encoders | MIT |
| h5py | demonstration storage | BSD-3-Clause |
| imageio, imageio-ffmpeg (FFmpeg) | MP4 writing | BSD-2-Clause; FFmpeg under LGPL/GPL depending on the build |
| OpenCV (`opencv-python`) | text and marker overlays on videos | Apache-2.0 |
| NumPy, PyYAML, TensorBoard | numerics, configuration, training logs | BSD-3-Clause; MIT; Apache-2.0 |

The CLIP weights are downloaded at first use from the Hugging Face Hub into the user's cache
(`~/.cache/huggingface`). Their MIT license requires that its copyright and permission notice accompany copies
of the weights; this repository does not include the weights themselves.
