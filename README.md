# PanoFormer
:triangular_flag_on_post::triangular_flag_on_post::triangular_flag_on_post:This is the pytorch implementation of PanoFormer (PanoFormer: Panorama Transformer for Indoor 360° Depth Estimation, ECCV 2022)!  
# Methodology
<table>
	<tr><th colspan="2" width="1200" height="200"><center><img src="/img/concept.png">The motivation of PanoFormer.</center></th></tr>
	<tr><th colspan="2" width="1000" height="200"><center><img src="/img/PanoFormer.png">Framework of PanoFormer.</center></th></tr>
    <tr>
        <td ><center><img src="/img/stlm.png" >STLM</center></td>
        <td ><center><img src="/img/psa.png">PSA</center></td>
    </tr>
</table>



# Datasets
You should download and prepare the datasets from the official webpage, *[Stanford2D3D](http://buildingparser.stanford.edu/dataset.html#Download)*, *[Matterport3D](https://niessner.github.io/Matterport/)*, *[3D60](https://vcl3d.github.io/3D60/)*. Unfortunately, PanoSunCG is not avaliable now. For Matterport3D. please follow the processing strategy in *[Unifuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion/blob/main/UniFuse/Matterport3D/README.md)*.  
:disappointed_relieved:*Attention*  
The version of *[Stanford2D3D and Matterport3D](https://zenodo.org/record/3492155#.YteQ1flBxPZ)* that contained in *[3D60](https://vcl3d.github.io/3D60/)* have a problem: The processed Matterport3D and Stanford2D3D leak the depth information via pixel brightness (pointed by our reviewers, we want to list it to alert the subsquent researchers).  
:heart:*Recommendation datasets*  
*[Stanford2D3D](http://buildingparser.stanford.edu/dataset.html#Download)*,*[Pano3D](https://vcl3d.github.io/Pano3D/download/)*(with a solid baseline), and *[Structured3D](https://structured3d-dataset.org/)*  
# Metrics  
For calculating MAE and MRE, please refer to *[SliceNet](https://github.com/crs4/SliceNet/blob/main/misc/eval.py)*. For others, please refer to *[Unifuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion/blob/main/UniFuse/metrics.py)*.  
:heart:*New metrics*  
Coming soon.
# Requirements
# Acknowledgements
We thank the authors of the projects below:  
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
