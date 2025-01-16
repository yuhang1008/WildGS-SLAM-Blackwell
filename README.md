<p align="center">

  <h1 align="center">WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments</h1>
  <p align="center">
    <a href="https://jianhao-zheng.github.io/"><strong>Jianhao Zheng*</strong></a>
    .
    <a href="https://zzh2000.github.io"><strong>Zihan Zhu*</strong></a>
    ·
    <a href="https://www.linkedin.com/in/valentin-bieri-98426b207/?originalSubdomain=ch"><strong>Valentin Bieri</strong></a>
    .
    <a href="https://people.inf.ethz.ch/pomarc/"><strong>Marc Pollefeys</strong></a>
    ·
    <a href="https://pengsongyou.github.io"><strong>Songyou Peng</strong></a>
    ·
    <a href="https://ir0.github.io/"><strong>Iro Armeni</strong></a>
</p>
  <h3 align="center">(Link to be added)<a href="no_link">Paper</a> | <a href="no_link">Video</a> | <a href="no_link">Project Page</a></h3>
  <div align="center"></div>
</p>
<p align="center">
    <img src="./media/teaser.png" alt="teaser_image" width="100%">
</p>

<p align="center">
Given a monocular video sequence captured in the wild with dynamic distractors,
WildGS-SLAM accurately tracks the camera trajectory and reconstructs a 3D Gaussian map for static elements, effectively removing all dynamic components. 
</p>
<br>



<br>
<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#quick-demo">Quick Demo</a>
    </li>
    <li>
      <a href="#run">Run</a>
    </li>
    <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>


## Installation

1. First you have to make sure that you clone the repo with the `--recursive` flag.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 
```bash
git clone --recursive https://github.com/GradientSpaces/WildGS-SLAM.git
cd WildGS-SLAM
```
2. Creating a new conda environment. 
```bash
conda create --name wildgs-slam python=3.10
conda activate wildgs-slam
```
3. Install CUDA 11.7 using conda and pytorch 1.12
```bash
conda install conda-forge::cudatoolkit-dev=11.7.0
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
5. Install the remaining dependencies.
```bash
python -m pip install -e thirdparty/lietorch/
python -m pip install -e thirdparty/diff-gaussian-rasterization-w-pose/
python -m pip install -e thirdparty/simple-knn/
```
6. Check installation.
```bash
python -c "import torch; import lietorch; import simple_knn; import
diff_gaussian_rasterization; print(torch.cuda.is_available())"
```
7. Now install the droid backends and the other requirements
```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

8. Download pretrained model.
Download the pretained models [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing), put it inside the `pretrained` folder.

## Quick Demo
First download the crowd sequence of Wild-SLAM dataset from (todo: add the link when the data is available)
```
python run.py  ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo.yaml
```

## Run

### Bonn Dynamic Dataset
Download the data as below and the data is saved into the `./Datasets/Bonn` folder. Note that the script only downloads the 8 sequences reported in the paper. To get other sequences, you can download from the [webiste of Bonn Dynamic Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html).
```bash
bash scripts/download_bonn.sh
```
You can run WildGS-SLAM via the following command:
```bash
python run.py  ./configs/Dynamic/Bonn/{config_file}
```
We have prepared config files for the 8 sequences. Note that this dataset needs preprocess to transform the pose. We have implemented that in the dataloader. If you want to test with sequences other than the ones provided, don't forget to specify ```dataset: 'bonn_dynamic'``` in your config file. The easiest way is to inherit from ```bonn_dynamic.yaml```.

### TUM RGB-D (dynamic) Dataset
Download the data (9 dynamic sequences) as below and the data is saved into the `./Datasets/TUM_RGBD` folder. 
```bash
bash scripts/download_tum.sh
```
The config files for 9 dynamic sequences of this dataset can be found under ```./configs/Dynamic/TUM_RGBD```. You can run WildGS-SLAM as the following:
```bash
python run.py  ./configs/Dynamic/TUM_RGBD/{config_file}
```

### Wild-SLAM dataset
To-do (add more here when it can be released)

## Evaluation

### Camera poses
The camera trajectories will be automatically evaluated after each run of WildGS-SLAM (if GT pose is provided). Statistics of the results are summarized in ```{save_dir}/traj/metrics_full_traj.txt```. The estimated camera poses are saved in ```{save_dir}/traj/est_poses_full.txt``` following the [TUM format](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).

### Novel View Synthesis
Only support for Wild-SLAM Mocap dataset. Write more when the dataset is ready to be public.

## Acknowledgement
We adapted some codes from some awesome repositories including [MonoGS](https://github.com/muskie82/MonoGS), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [Splat-SLAM](https://github.com/google-research/Splat-SLAM), [GIORIE-SLAM](https://github.com/zhangganlin/GlORIE-SLAM), [nerf-on-the-go](https://github.com/cvg/nerf-on-the-go) and [Metric3D V2](https://github.com/YvanYin/Metric3D). Thanks for making codes public available. 

## Citation

If you find our code or paper useful, please cite
```bibtex
@inproceedings{add this when arXiv is available (todo)}
}
```

## Contact
Contact [Jianhao Zheng](mailto:jianhao@stanford.edu) for questions, comments and reporting bugs.