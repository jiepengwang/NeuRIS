# NeuRIS
We propose a new method, dubbed NeuRIS, for high quality reconstruction of indoor scenes. 

![](./doc/teaser.png)

## [Project page](https://jiepengwang.github.io/NeuRIS/) |  [Paper](https://arxiv.org/abs/2206.13597) | [Data]


### Setup
```
conda create -n neuris python=3.8
conda activate neuris
conda install pytorch=1.9.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```


### Training

```
python ./exp_runner.py --mode train --conf ./confs/neuris.conf --gpu 0 --scene_name scene0625_00
```

### TODO
* Add evaluation code
* Add data preprocessing code


## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{wang2022neuris,
      	title={NeuRIS: Neural Reconstruction of Indoor Scenes Using Normal Priors}, 
      	author={Wang, Jiepeng and Wang, Peng and Long, Xiaoxiao and Theobalt, Christian and Komura, Taku and Liu, Lingjie and Wang, Wenping},
	publisher = {arXiv},
      	year={2022}
}
```