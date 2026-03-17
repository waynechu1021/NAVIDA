<div align="center">
  <h1 style="font-size: 30px; font-weight: bold;"> NaVIDA: Vision-Language Navigation with Inverse Dynamics Augmentation</h1>

  <br>

  <a href="https://arxiv.org/abs/2601.18188">
    <img src="https://img.shields.io/badge/ArXiv-2601.18188-brown?logo=arxiv" alt="Paper">
  </a>
  <a href="https://huggingface.co/waynechu/NaVIDA">
    <img src="https://img.shields.io/badge/🤗 huggingface-Model-blue" alt="checkpoint">
  </a>
</div>

## NaVIDA

NaVIDA is a lightweight VLN framework that incorporates inverse dynamics supervision as an explicit objective to embed action-grounded visual dynamics into policy learning. We employs hierarchical probabilistic action chunking to organizes trajectories into multi-step chunks to support 


## 🛠 Getting Started

### Setup the Environemnt

**Create Env**  
```bash
conda create -n navida python=3.10
conda activate navida
```
**Install habitat-sim v0.2.4**
```bash
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless
```
**Install habitat-lab v0.2.4**
```bash
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd ..
cd habitat-lab
pip install -e habitat-lab  # install habitat_lab
pip install -e habitat-baselines # install habitat_baselines
pip install dtw fastdtw gym
```
**Install package for NaVIDA**
```
pip install peft trl==0.16.0 transformers==4.50.3 tensorboardx qwen_vl_utils deepspeed distilabel wandb==0.18.3
pip install numpy==1.24.0 numba==0.60.0
pip install vllm==0.9.1 torch torchvision protobuf==3.20
pip install flash-attn --no-build-isolation --no-cache-dir
```

### Data Preparation

- 1. **Scene Datasets**  
   - For **R2R**, **RxR** and **EnvDrop**: Download the MP3D scenes from the [official project page](https://niessner.github.io/Matterport/), and place them under `data/scene_datasets/mp3d/`.
   - For **ScaleVLN**: Download the HM3D scenes from the [official github page](https://github.com/matterport/habitat-matterport-3dresearch), and place the `train` split under `data/scene_datasets/hm3d/`
- 2. **Download Data**
  - VLN-CE Episodes: 
    - [R2R](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view)
    - [RxR](https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view)
  - ScaleVLN: Download the the preprocessed subset of ScaleVLN from [StreamVLN](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data/blob/main/ScaleVLN/scalevln_subset_150k.json.gz), which has been converted to the VLN-CE format.
- 3. **Data Precess (For Training Only)**

  - Convert all the data you download above (R2R, RxR, EnvDrop, ScaleVLN) into a unified format.
  ```bash
  ./scripts/preprocess.sh
  ```
  - Extrac the RGB frame
  ```bash
  ./scripts/extract_frame.sh
  ```
  - Prepare the Training data
  ```bash
  ./scripts/prepare_training_data.sh
  ```
  You may also prepare your DAgger in the same format.


## 🔥 Training
```bash
./scripts/train.sh
```


## 🧭 Evaluation

### Eval with HuggingFace Transformers
```bash
./scripts/eval.sh
```

### Eval with vLLM
Using vllm for faster inference. First you need launch the server
```bash
./scripts/start_vllm_server.sh
```
Then start the evaluation scripts like
```bash
./scripts/eval_vllm.sh
```

## 🏆 Checkpoints
We provide the [`checkpoints`](https://huggingface.co/Arvil/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn) in HuggingFace for benchmark reproduction.



## 🔗 Citation

If you find our work helpful, please consider starring this repo 🌟 and cite:

```bibtex
@article{zhu2026textsc,
  title={$\backslash$textsc $\{$NaVIDA$\}$: Vision-Language Navigation with Inverse Dynamics Augmentation},
  author={Zhu, Weiye and Zhang, Zekai and Wang, Xiangchen and Pan, Hewei and Wang, Teng and Geng, Tiantian and Xu, Rongtao and Zheng, Feng},
  journal={arXiv preprint arXiv:2601.18188},
  year={2026}
}
```

## 👏 Acknowledgements

We would like to thank the authors of [NaVid](https://github.com/jzhzhang/NaVid-VLN-CE), [StreamVLN](https://github.com/InternRobotics/StreamVLN) for their great works.