# Iterative Vision-and-Language Navigation in Continuous Environments (IVLN-CE)

[Jacob Krantz*](https://jacobkrantz.github.io), [Shurjo Banerjee*](https://shurjobanerjee.github.io), [Wang Zhu](https://billzhu.me), [Jason Corso](https://web.eecs.umich.edu/~jjcorso), [Peter Anderson](https://panderson.me), [Stefan Lee](http://web.engr.oregonstate.edu/~leestef), and [Jesse Thomason](https://jessethomason.com)

[[Project Page](https://jacobkrantz.github.io/ivln)] [[Paper](http://arxiv.org/abs/2210.03087)] [[IVLN Code](https://github.com/Bill1235813/IVLN)]

This is the official implementation of **Iterative Vision-and-Language Navigation (IVLN) in continuous environments**, a paradigm for evaluating language-guided agents navigating in a persistent environment over time. Existing Vision-and-Language Navigation (VLN) benchmarks erase the agent’s memory at the beginning of every episode, testing the ability to perform cold-start navigation with no prior information. However, deployed robots occupy the same environment for long periods of time. The IVLN paradigm addresses this disparity by training and evaluating VLN agents that maintain memory across tours of scenes that consist of up to 100 ordered instruction-following Room-to-Room (R2R) episodes each defined by an individual language instruction and a target path. This repository implements the **Iterative Room-to-Room in Continuous Environments (IR2R-CE)** benchmark.

<p align="center">
  <img width="606" height="324" src="./data/res/ivln.png" alt="IVLN">
</p>

## Setup

This project is modified from the [VLN-CE](https://github.com/jacobkrantz/VLN-CE) repository starting from [this commit](https://github.com/jacobkrantz/VLN-CE/tree/4ef91c0501862026e91a6d7177e515cb043575a4).

1. Initialize the project

```bash
git clone --recurse-submodules git@github.com:jacobkrantz/Iterative-VLNCE.git
cd Iterative-VLNCE

conda env create -f environment.yml
conda activate ivlnce
```

Note: if you have runtime issues relating to [torch-scatter](https://github.com/rusty1s/pytorch_scatter), reinstall it with the cuda-supported wheel. In my case, this was:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
```

2. Download the Matterport3D scene meshes

```bash
# run with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
# Extract to: ./data/scene_datasets/mp3d/{scene}/{scene}.glb
```

`download_mp.py` must be obtained from the Matterport3D [project webpage](https://niessner.github.io/Matterport/).

3. Download the Room-to-Room episodes in VLN-CE format ([link](https://drive.google.com/file/d/1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm/view))

```bash
gdown https://drive.google.com/uc?id=1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm
# Extract to: ./data/datasets/R2R_VLNCE_v1-3/{split}/{split}.json.gz
```

4. Download files that define tours of episodes:

| Weights | Download | Extract Path |
|-|-|-|
| Tour ordering | [Link](https://drive.google.com/file/d/1aeMFpDyabpb7nTL-MGCLKsab-otE8Niu/view?usp=sharing) (1 MB) | `data/tours.json` |
| Target paths for t-nDTW eval | [Link](https://drive.google.com/file/d/1zxApwFsUF0ekPCiB6z0WQLKNBIVdAgOR/view?usp=sharing) (132 MB) | `data/gt_ndtw.json` |

5. [OPTIONAL] To run baseline models, the following weights are required:

| Weights | Download | Extract Path |
|-|-|-|
|ResNet Depth Encoder (DDPPO-trained) | [Link](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7/habitat_baselines/rl/ddppo) (745 MB) | `data/ddppo-models/{model}.pth` |
| Semantics inference (RedNet) | [Link](https://drive.google.com/file/d/15SWMgLxVhhTitYr55RDeEp4W8zQ7HkOq/view?usp=sharing) (626 MB) | `data/rednet_mp3d_best_model.pkl` |
| Pre-trained MapCMA models | [Link](https://drive.google.com/drive/folders/1JrUz-ScwtxxKVhXtNeglui16Rs1V_7A7?usp=sharing) (608 MB) | `data/checkpoints/{model}.pth` |
| Pre-computed known maps |[Link](https://drive.google.com/file/d/1sLCFjGfdRuyMqSIQRQwWew-IvNmqFNcT/view?usp=sharing) (78 MB)| `data/known_maps/{semantic-src}/{scene}.npz`|

## Starter Code

The `run.py` script controls training and evaluation for all models:

```bash
python run.py \
  --exp-config path/to/experiment_config.yaml \
  --run-type {train | eval}
```

Config files exist for running each experiment detailed in the paper, both for training and for evaluation. The configs for running ground-truth semantics experiments are located in `ivlnce_baselines/config/map_cma/gt_semantics` and the configs for running predicted semantics experiments are located in `ivlnce_baselines/config/map_cma/pred_semantics`. Each subfolder `{episodic, iterative, known}` contains configs for training and evaluating a model with that mapping method. Following the numbered order of config `.yaml` files in each respective directory will train the model and evaluate it on all mapping modes. The unstructured memory models are represented in the `ivlnce_baselines/config/latent_baselines` folder.

### Evaluating Pre-trained MapCMA Models

The naming convention of pre-trained MapCMA models is `[semantics]_[training].pth` where `semantics` is either gt (ground-truth) or pred (predicted from RedNet) and `training` is the map construction method: either episodic (ep), iterative (it), or known (kn). Each can be evaluated with existing config files. For example, consider a model trained on predicted semantics and with iterative maps (`pred_it.pth`). To evalaute this model in the same setting, run:

```bash
python run.py \
  --run-type eval \
  --exp-config ivlnce_baselines/config/map_cma/pred_semantics/iterative_maps/2_eval_iterative.yaml \
  EVAL_CKPT_PATH_DIR data/checkpoints/pred_it.pth
```

Similarly, this model can be evaluated with known maps:

```bash
python run.py \
  --run-type eval \
  --exp-config ivlnce_baselines/config/map_cma/pred_semantics/iterative_maps/2_eval_iterative.yaml \
  EVAL_CKPT_PATH_DIR data/checkpoints/pred_it.pth
```

You can look through the configs in `ivlnce_baselines/config/map_cma` to find a particular training or evaluation configuration of interest.

### Training Agents
The `DaggerTrainer` class is the standard trainer and supports teacher forcing or dataset aggregation (DAgger) of episodic data. We also include the `IterativeCollectionDAgger` trainer which builds maps iteratively and then trains agents episodically on those maps. The `IterativeDAggerTrainer` collects and trains models iteratively and is used to train unstructured memory models on IR2R-CE. All trainers inherit from `BaseVLNCETrainer`.

#### Training MapCMA

Suppose you want to train a MapCMA model from scratch with predicted semantics and iterative maps, like was done in the paper. First, train on IR2R-CE + augmented tour data using teacher forcing:

```bash
python run.py \
  --run-type train \
  --exp-config ivlnce_baselines/config/map_cma/pred_semantics/iterative_maps/0_train_tf.yaml
```

Then, swap `train` for `eval` to evaluate each checkpoint. Take the best performing checkpoint and fine-tune with DAgger on the IR2R-CE tours:

```bash
python run.py \
  --run-type train \
  --exp-config ivlnce_baselines/config/map_cma/pred_semantics/iterative_maps/1_ftune_dagger.yaml \
  IL.ckpt_to_load path/to/best/checkpoint.pth
```

Finally, evaluate each resulting checkpoint to find the best on the `val_unseen` split:

```bash
python run.py \
  --run-type eval \
  --exp-config ivlnce_baselines/config/map_cma/pred_semantics/iterative_maps/2_eval_iterative.yaml
```

While this tutorial walked through a single example, config sequences are provided for all models in the paper (both latent CMA and MapCMA).

## Citation

If you find this work useful, please consider citing:

```
@article{krantz2022iterative
  title={Iterative Vision-and-Language Navigation},
  author={Krantz, Jacob and Banerjee, Shurjo and Zhu, Wang and Corso, Jason and Anderson, Peter and Lee, Stefan and Thomason, Jesse},
  journal={arXiv preprint arXiv:2210.03087},
  year={2022},
}
```

## License

This codebase is [MIT licensed](LICENSE). Trained models and task datasets are considered data derived from the mp3d scene dataset. Matterport3D based task datasets and trained models are distributed with [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).

