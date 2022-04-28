# [Ditto](https://ut-austin-rpl.github.io/Ditto/) data generation

This repo stores the codes for data generation of *Ditto: Building Digital Twins of Articulated Objects from Interaction*. Given URDF file of an articulated object, we spwan it in Pybullet simulation, and interact with it by directly manipulating the joint state. We collect multiview depth observations of the object before and after interaction, as well as the ground truth occupancy data and segmentation.

## Installation

1. `pip install -r requirements.txt`
2. `pip install -e .`
3. `python scripts/convonet_setup.py build_ext --inplace`

## URDF preparation

Download Shape2Motion\_urdfs.zip from [here](https://utexas.box.com/s/dx8kanv7zs7rdmu7kehc804vsp2gw25z), unzip it and put it under `data/urdfs`. If you use this data, remember to cite [Shape2Motion](https://arxiv.org/abs/1903.03911) paper.

## Generate data

You can either directly run `scripts/generate_data_s2m.sh` or run the commands in that bash file.

## Citation

If you find this repo useful, please cite

```
@inproceedings{jiang2022ditto,
   title={Ditto: Building Digital Twins of Articulated Objects from Interaction},
   author={Jiang, Zhenyu and Hsu, Cheng-Chun and Zhu, Yuke},
   booktitle={arXiv preprint arXiv:2202.08227},
   year={2022}
}
```
