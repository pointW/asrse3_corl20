# Augmented State Representation in SE(3) Action Spaces
This repository contains the code of the paper [Policy learning in SE(3) action spaces](https://arxiv.org/abs/2010.02798). Project website: https://pointw.github.io/asrse3-page/

## Installation
1. Install [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
1. Clone this repo
    ```
    git clone https://github.com/pointW/asrse3_corl20.git
    cd asrse3_corl20
    ```
1. Create and activate conda environment, install requirement packages
    ```
    conda create --name asrse3 python=3.7
    conda activate asrse3
    pip install -r requirements.txt
    cd ..
    ```
    Note that this project was developed under pybullet version 2.7.1. Newer version of pybullet should also work, but it is not tested.   
1. Install [PyTorch](https://pytorch.org/) (Recommended: pytorch==1.7.0, torchvision==0.8.1)
1. Install [CuPy](https://github.com/cupy/cupy)
1. Clone and install the environment repo
    ```
    git clone https://github.com/ColinKohler/helping_hands_rl_envs.git -b dian_corl20
    cd helping_hands_rl_envs
    pip install .
    cd ..
    ```
1. Goto the scripts folder of this repo to run experiments
    ```
    cd asrse3_corl20/scripts
    ```
    
## 3D (x y theta) Experiments
### Deconstruction Data Collection
#### Example: 5H1
```
python fill_buffer_deconstruct.py --num_process=20 --alg=margin_asr --action_sequence=xyrp --buffer_size=50000 --env=house_building_1_deconstruct --num_objects=5 --max_episode_steps=10 --log_sub=h1_deconstruct 
```

#### Other Envs
Replace the following parameters for `fill_buffer_deconstruct.py`: `--env=house_building_1_deconstruct --num_objects=5 --max_episode_steps=10 --log_sub=h1_deconstruct`
into:

* H4: `--env=house_building_4_deconstruct --num_objects=6 --max_episode_steps=20 --log_sub=h4_deconstruct`
* ImDis: `--env=improvise_house_building_discrete_deconstruct --num_objects=5 --max_episode_steps=10 --log_sub=imdis_deconstruct`
* ImRan: `--env=improvise_house_building_random_deconstruct --num_objects=5 --max_episode_steps=10 --log_sub=imran_deconstruct`

### Training
#### Example: training 5H1 using ASRSE3 SDQfD
```
python main.py --num_process=5 --alg=margin_asr --action_sequence=xyrp --planner_episode=0 --explore=0 --fixed_eps --buffer=expert --max_episode=50000 --pre_train_step=10000 --env=house_building_1 --num_objects=5 --max_episode_steps=10 --load_buffer=outputs/margin_asr_deconstruct/h1_deconstruct/checkpoint/buffer.pt 
```
#### Other envs:
Replace the following parameters for `main.py`:
`--env=house_building_1 --num_objects=5 --max_episode_steps=10 --load_buffer=outputs/margin_asr_deconstruct/h1_deconstruct/checkpoint/buffer.pt`
with:

* H4: `--env=house_building_4 --num_objects=6 --max_episode_steps=20 --load_buffer=outputs/margin_asr_deconstruct/h4_deconstruct/checkpoint/buffer.pt`
* ImDis: `--env=improvise_house_building_discrete --num_objects=5 --max_episode_steps=10 --load_buffer=outputs/margin_asr_deconstruct/imdis_deconstruct/checkpoint/buffer.pt`
* ImRan: `--env=improvise_house_building_random --num_objects=5 --max_episode_steps=10 --load_buffer=outputs/margin_asr_deconstruct/imran_deconstruct/checkpoint/buffer.pt`

#### Other algorithms
Replace `--alg=margin_asr` in `main.py` with:
- ASRSE3 DQfD: `--alg=margin_asr --margin=oril`
- ASRSE3 ADET: `--alg=margin_asr --margin=ce`
- ASRSE3 DQN: `--alg=dqn_asr`
- FCN SDQfD: `--alg=margin_fcn`
- FCN DQfD: `--alg=margin_fcn --margin=oril`
- FCN ADET: `--alg=margin_fcn --margin=ce --margin_weight=0.01`
- FCN DQN: `--alg=dqn_fcn`

## 6D Experiments

### Deconstruction Data Collection
#### Example: 4H1
```
python fill_buffer_deconstruct.py --num_process=20 --alg=margin_asr_5l --action_sequence=xyzrrrp --in_hand_mode=proj --buffer_size=50000 --env=ramp_house_building_1_deconstruct --num_objects=4 --max_episode_steps=10 --log_sub=ramp_h1_deconstruct
```

#### Other Envs
Replace the following parameters for `fill_buffer_deconstruct.py`:
`--env=ramp_house_building_1_deconstruct --num_objects=4 --max_episode_steps=10 --log_sub=ramp_h1_deconstruct`
with:

* H3: `--env=ramp_house_building_3_deconstruct --num_objects=4 --max_episode_steps=10 --log_sub=ramp_h3_deconstruct`
* H4: `--env=ramp_house_building_4_deconstruct --num_objects=6 --max_episode_steps=20 --log_sub=ramp_h4_deconstruct`
* ImH2: `--env=ramp_improvise_house_building_2_deconstruct --num_objects=3 --max_episode_steps=10 --log_sub=ramp_imh2_deconstruct`
* ImH3: `--env=ramp_improvise_house_building_3_deconstruct --num_objects=4 --max_episode_steps=10 --log_sub=ramp_imh3_deconstruct`

### Training
#### Example: 4H1
```
python main.py --num_process=5 --alg=margin_asr_5l --action_sequence=xyzrrrp --in_hand_mode=proj --planner_episode=0 --explore=0 --fixed_eps --buffer=expert --max_episode=50000 --pre_train_step=10000 --env=ramp_house_building_1 --num_objects=4 --max_episode_steps=10 --load_buffer=outputs/margin_asr_5l_deconstruct/ramp_h1_deconstruct/checkpoint/buffer.pt
```

#### Other envs:
Replace the following parameters for `main.py`:
`--env=ramp_house_building_1 --num_objects=4 --max_episode_steps=10 --load_buffer=outputs/margin_asr_5l_deconstruct/ramp_h1_deconstruct/checkpoint/buffer.pt`
into:
* H3: `--env=ramp_house_building_3 --num_objects=4 --max_episode_steps=10 --load_buffer=outputs/margin_asr_5l_deconstruct/ramp_h3_deconstruct/checkpoint/buffer.pt`
* H4: `--env=ramp_use_building_4 --num_objects=6 --max_episode_steps=20 --load_buffer=outputs/margin_asr_5l_deconstruct/ramp_h4_deconstruct/checkpoint/buffer.pt`
* ImH2: `--env=ramp_improvise_house_building_2 --num_objects=3 --max_episode_steps=10 --load_buffer=outputs/margin_asr_5l_deconstruct/ramp_imh2_deconstruct/checkpoint/buffer.pt`
* ImH3: `--env=ramp_improvise_house_building_3 --num_objects=4 --max_episode_steps=10 --load_buffer=outputs/margin_asr_5l_deconstruct/ramp_imh3_deconstruct/checkpoint/buffer.pt`

#### Other algorithms
Replace `--alg=margin_asr_5l` in `main.py` with:
- ASRSE3 DQfD: `--alg=margin_asr_5l --margin=oril`
- ASRSE3 ADET: `--alg=margin_asr_5l --margin=ce --margin_weight=0.01`
- ASRSE3 DQN: `--alg=dqn_asr_5l`

## Results
The training results will be saved under `scripts/outputs`

## Citation
```
@article{wang2020policy,
  title={Policy learning in SE (3) action spaces},
  author={Wang, Dian and Kohler, Colin and Platt, Robert},
  journal={arXiv preprint arXiv:2010.02798},
  year={2020}
}
```