# PanoFormer
This is the pytorch implementation of PanoFormer!  
# Methodology
<img src="/img/concept.png">
<img src="/img/stlm.png" height="80%">
<img src="/img/PanoFormer.png" width="50%">

# Datasets
You should download and prepare the datasets from the official webpage, *[Stanford2D3D](http://buildingparser.stanford.edu/dataset.html#Download)*, *[Matterport3D](https://niessner.github.io/Matterport/)*, *[3D60](https://vcl3d.github.io/3D60/)*. Unfortunately, PanoSunCG is not avaliable now. For Matterport3D. please follow the processing strategy in *[Unifuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion/blob/main/UniFuse/Matterport3D/README.md)*.
# Metrics
# Requirements
# Acknowledgements
We thank the authors of the codes below:  
*[Unifuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion)*, *[Uformer](https://github.com/ZhendongWang6/Uformer)*, *[SphereNet](https://github.com/mty1203/spherenet)*, *[DeformableAtt](https://github.com/yhy258/Deformable_Attention__DeformableDETR)*  
If you find our work useful, please consider citing： 
```
@article{shen2022panoformer,
  title={PanoFormer: Panorama transformer for indoor 360 depth estimation},
  author={Shen, Zhijie and Lin, Chunyu and Liao, Kang and Nie, Lang and Zheng, Zishuo and Zhao, Yao},
  journal={arXiv e-prints},
  pages={arXiv--2203},
  year={2022}
}
```
```
@inproceedings{shen2021distortion,
  title={Distortion-tolerant monocular depth estimation on omnidirectional images using dual-cubemap},
  author={Shen, Zhijie and Lin, Chunyu and Nie, Lang and Liao, Kang and Zhao, Yao},
  booktitle={2021 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```
And also these:  
```
@article{jiang2021unifuse,
      title={UniFuse: Unidirectional Fusion for 360$^{\circ}$ Panorama Depth Estimation}, 
      author={Hualie Jiang and Zhe Sheng and Siyu Zhu and Zilong Dong and Rui Huang},
	  journal={IEEE Robotics and Automation Letters},
	  year={2021},
	  publisher={IEEE}
}
```
```
@InProceedings{Wang_2022_CVPR,
    author    = {Wang, Zhendong and Cun, Xiaodong and Bao, Jianmin and Zhou, Wengang and Liu, Jianzhuang and Li, Houqiang},
    title     = {Uformer: A General U-Shaped Transformer for Image Restoration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {17683-17693}
}
```
```
@article{zhu2020deformable,
  title={Deformable detr: Deformable transformers for end-to-end object detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
```
