<h2 align="center">SymPoint Revolutionized: Boosting Panoptic Symbol Spotting with Layer Feature Enhancement</h2>
<p align="center">
  <img src="assets/framework.png" width="75%">
</p>



## 🔧Installation & Dataset
#### Environment

We recommend users to use `conda` to install the running environment. The following dependencies are required:

```bash
conda create -n spv1 python=3.8 -y
conda activate spv1

pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install gdown mmcv==0.2.14 svgpathtools==1.6.1 munch==2.5.0 tensorboard==2.12.0 tensorboardx==2.5.1 detectron2==0.6
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# compile pointops
cd modules/pointops
python setup.py install
```

#### Dataset&Preprocess

download dataset from floorplan website, and convert it to json format data for training and testing.

```python
# download dataset
python download_data.py
# preprocess
#train, val, test
python parse_svg.py --split train --data_dir ./dataset/train/train/svg_gt/
python parse_svg.py --split val --data_dir ./dataset/val/val/svg_gt/
python parse_svg.py --split test --data_dir ./dataset/test/test/svg_gt/
```

## 🚀Quick Start

```
#train
bash tools/train_dist.sh
#test
bash tools/test_dist.sh
```

## 📌Citation
If you find our paper and code useful in your research, please consider giving a star and citation.
<pre><code>
@article{liu2024sympoint,
  title={SymPoint Revolutionized: Boosting Panoptic Symbol Spotting with Layer Feature Enhancement},
  author={Liu, Wenlong and Yang, Tianyu and Yu, Qizhi and Zhang, Lei},
  journal={arXiv preprint arXiv:2407.01928},
  year={2024}
}
</code></pre>
