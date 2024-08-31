# CSTA: CNN-based Spatiotemporal Attention for Video Summarization (CVPR 2024 paper)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/csta-cnn-based-spatiotemporal-attention-for/supervised-video-summarization-on-summe)](https://paperswithcode.com/sota/supervised-video-summarization-on-summe?p=csta-cnn-based-spatiotemporal-attention-for) <br/>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/csta-cnn-based-spatiotemporal-attention-for/supervised-video-summarization-on-tvsum)](https://paperswithcode.com/sota/supervised-video-summarization-on-tvsum?p=csta-cnn-based-spatiotemporal-attention-for) <br/>

The official code of "[CSTA: CNN-based Spatiotemporal Attention for Video Summarization](https://openaccess.thecvf.com/content/CVPR2024/papers/Son_CSTA_CNN-based_Spatiotemporal_Attention_for_Video_Summarization_CVPR_2024_paper.pdf)" <br/>
![image](https://github.com/thswodnjs3/CSTA/assets/93433004/aa0dff4d-9b29-49a2-989a-5b6a12dba5fe)

 * [Model overview](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#model-overview)
 * [Updates](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#updates)
 * [Requirements](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#requirements)
 * [Data](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#data)
 * [Pre-trained models](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#pre-trained-models)
 * [Training](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#training)
 * [Inference](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#inference)
 * [Generate summary videos](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#generate-summary-videos)
 * [Citation](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#citation)
 * [Acknowledgement](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#acknowledgement)

# Model overview
![image](https://github.com/thswodnjs3/CSTA/assets/93433004/537b7375-10d7-4d7d-8de0-0b69631ac635) <br/>
<br/>
[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑

# Updates
 * [2024.03.24] Create a repository.
 * [2024.05.21] Update the code and pre-trained models.
 * [2024.07.18] Upload the code to generate summary videos, including custom videos.
 * [2024.07.21] Update the KTS code for full frames of videos.
 * [2024.07.23] Update the code to use only the CPU.
 * (Yet) [2024.09.??] Add detailed explanations and comments for the code.

[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑

# Requirements
|Ubuntu|GPU|CUDA|cuDNN|conda|python|
|:---:|:---:|:---:|:---:|:---:|:---:|
|20.04.6 LTS|NVIDIA GeForce RTX 4090|12.1|8902|4.9.2|3.8.5|

|h5py|numpy|scipy|torch|torchvision|tqdm|
|:---:|:---:|:---:|:---:|:---:|:---:|
|3.1.0|1.19.5|1.5.2|2.2.1|0.17.1|4.61.0|

```
conda create -n CSTA python=3.8.5
conda activate CSTA
git clone https://github.com/thswodnjs3/CSTA.git
cd CSTA
pip install -r requirements.txt
```

[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑

# Data
Link: [Dataset](https://drive.google.com/drive/folders/1iGfKZxexQfOxyIaOWhfU0P687dJq_KWF?usp=drive_link) <br/>
H5py format of two benchmark video summarization preprocessed datasets (SumMe, TVSum). <br/>
You should download datasets and put them in ```data/``` directory. <br/>
The structure of the directory must be like below. <br/>
```
 ├── data
     └── eccv16_dataset_summe_google_pool5.h5
     └── eccv16_dataset_tvsum_google_pool5.h5
```
You can see the details of both datasets below. <br/>

[SumMe](https://link.springer.com/chapter/10.1007/978-3-319-10584-0_33) <br/>
[TVSum](https://openaccess.thecvf.com/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf) <br/>
<br/>
[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑

# Pre-trained models
Link: [Weights](https://drive.google.com/drive/folders/1Z0WV_IJAHXV16sAGW7TmC9J_iFZQ9NSs?usp=drive_link) <br/>
You can download our pre-trained weights of CSTA. <br/>
There are 5 weights for the SumMe dataset and the other 5 for the TVSum dataset(1 weight for each split). <br/>
As shown in the paper, we tested everything 10 times (without fixation of seed) but only uploaded a single model as a representative for your convenience. <br/>
The uploaded weight is acquired when the seed is 123456, and the result is almost identical to our paper. <br/>
You should put 5 weights of the SumMe in ```weights/SumMe``` and the other 5 weights of the TVSum in ```weights/TVSum```. <br/>
The structure of the directory must be like below. <br/>
```
 ├── weights
     └── SumMe
         ├── split1.pt
         ├── split2.pt
         ├── split3.pt
         ├── split4.pt
         ├── split5.pt
     └── TVSum
         ├── split1.pt
         ├── split2.pt
         ├── split3.pt
         ├── split4.pt
         ├── split5.pt
```

[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑

# Training
You can train the final version of our models by command below. <br/>
```
python train.py
```
Detailed explanations for all configurations will be updated later. <br/>

## You can't reproduce our result perfectly.
As shown in the paper, we tested every experiment 10 times without fixation of the seed, so we can't be sure which seeds export the same results. <br/>
Even though you set the seed 123456, which is the same as our pre-trained models, it may result in different results due to the non-deterministic property of the [Adaptive Average Pooling layer](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms). <br/>
Based on my knowledge, non-deterministic operations produce random results even with the same seed. [You can see details here.](https://pytorch.org/docs/stable/notes/randomness.html) <br/>
However, you can get similar results with the pre-trained models when you set the seed as 123456, so I hope this will be helpful for you. <br/>
<br/>
[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑

# Inference
You can see the final performance of the models by command below. <br/>
```
python inference.py
```
All weight files should be located in the position I said above. <br/>
<br/>
[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑

# Generate summary videos
You can generate summary videos using our models. <br/>
You can use either videos from public datasets or custom videos. <br/>
With the code below, you can apply our pre-trained models to raw videos to produce summary videos. <br/>
```
python generate_video.py --input_is_file True or False
    --file_path 'path to input video'
    --dir_path 'directory of input videos'
    --ext 'video file extension'
    --save_path 'path to save summary video'
    --weight_path 'path to loaded weights'

e.g.
1)Using a directory
python generate_video.py --input_is_file False --dir_path './videos' --ext 'mp4' --save_path './summary_videos' --weight_path './weights/SumMe/split4.pt'

2)Using a single video file
python generate_video.py --input_is_file True --file_path './videos/Jumps.mp4' --save_path './summary_videos' --weight_path './weights/SumMe/split4.pt'
```
The explanation of the arguments is as follows. <br/>
If you change the 'ext' argument and input a directory of videos, you must modify the ['fourcc'](https://github.com/thswodnjs3/CSTA/blob/7227ee36a460b0bdc4aa83cb446223779365df45/generate_video.py#L34) variable in the 'produce_video' function within the 'generate_video.py' file. <br/>
Additionally, you must update this when inputting a single video file with different extensions other than 'mp4'.
```
1. input_is_file (bool): True or False
    Indicates whether the input is a file or a directory.
    If this is True, the 'file_path' argument is required.
    If this is False, the 'dir_path' and 'ext' arguments are required.

2. file_path (str) e.g. './SumMe/Jumps.mp4'
    The path of the video file.
    This is only used when 'input_is_file' is True.

3. dir_path (str) e.g. './SumMe'
    The path of the directory where video files are located.
    This is only used when 'input_is_file' is False.

4. ext (str) e.g. 'mp4'
    The file extension of the video files.
    This is only used when 'input_is_file' is False.

5. sample_rate (int) e.g. 15
    The interval between selected frames in a video.
    For example, if the video has 30 fps, it will become 2 fps with a sample_rate of 15.

6. save_path (str) e.g. './summary_videos'
    The path where the summary videos are saved.

7. weight_path (str) e.g. './weights/SumMe/split4.pt'
    The path where the model weights are loaded from.
```
We referenced the KTS code from [DSNet](https://github.com/li-plus/DSNet).<br/>
However, they applied KTS to downsampled videos (2 fps), which can result in different shot change points and sometimes make it impossible to summarize videos. <br/>
We revised it to calculate change points based on the entire frames. <br/>
<br/>
[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑

# Citation
If you find our code or our paper useful, please click [★star] for this repo and [cite] the following paper:
```
@inproceedings{son2024csta,
  title={CSTA: CNN-based Spatiotemporal Attention for Video Summarization},
  author={Son, Jaewon and Park, Jaehun and Kim, Kwangsu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18847--18856},
  year={2024}
}
```

[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑

# Acknowledgement
We especially, sincerely appreciate the authors of PosENet, RR-STG who responded to our requests very kindly. <br/>
Below are the papers we referenced for the code. <br/>

A2Summ - [paper](https://arxiv.org/pdf/2303.07284), [code](https://github.com/boheumd/A2Summ) <br/>
CA-SUM - [paper](https://www.iti.gr/~bmezaris/publications/icmr2022_preprint.pdf), [code](https://github.com/e-apostolidis/CA-SUM) <br/>
DSNet - [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9275314), [code](https://github.com/li-plus/DSNet) <br/>
iPTNet - [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Joint_Video_Summarization_and_Moment_Localization_by_Cross-Task_Sample_Transfer_CVPR_2022_paper.pdf) <br/>
MSVA - [paper](https://arxiv.org/pdf/2104.11530), [code](https://github.com/TIBHannover/MSVA) <br/>
PGL-SUM - [paper](https://www.iti.gr/~bmezaris/publications/ism2021a_preprint.pdf), [code](https://github.com/e-apostolidis/PGL-SUM) <br/>
PosENet - [paper](https://arxiv.org/pdf/2001.08248), [code](https://github.com/islamamirul/position_information) <br/>
RR-STG - [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9750933&tag=1) <br/>
SSPVS - [paper](https://arxiv.org/pdf/2201.02494), [code](https://github.com/HopLee6/SSPVS-PyTorch) <br/>
STVT - [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10124837), [code](https://github.com/nchucvml/STVT) <br/>
VASNet - [paper](https://arxiv.org/pdf/1812.01969), [code](https://github.com/ok1zjf/VASNet) <br/>
VJMHT - [paper](https://arxiv.org/pdf/2112.13478), [code](https://github.com/HopLee6/VJMHT-PyTorch) <br/>

```
@inproceedings{he2023a2summ,
  title = {Align and Attend: Multimodal Summarization with Dual Contrastive Losses},
  author={He, Bo and Wang, Jun and Qiu, Jielin and Bui, Trung and Shrivastava, Abhinav and Wang, Zhaowen},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
}
```
```
@inproceedings{10.1145/3512527.3531404,
  author = {Apostolidis, Evlampios and Balaouras, Georgios and Mezaris, Vasileios and Patras, Ioannis},
  title = {Summarizing Videos Using Concentrated Attention and Considering the Uniqueness and Diversity of the Video Frames},
  year = {2022},
  isbn = {9781450392389},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3512527.3531404},
  doi = {10.1145/3512527.3531404},
  pages = {407-415},
  numpages = {9},
  keywords = {frame diversity, frame uniqueness, concentrated attention, unsupervised learning, video summarization},
  location = {Newark, NJ, USA},
  series = {ICMR '22}
}
```
```
@article{zhu2020dsnet,
  title={DSNet: A Flexible Detect-to-Summarize Network for Video Summarization},
  author={Zhu, Wencheng and Lu, Jiwen and Li, Jiahao and Zhou, Jie},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={948--962},
  year={2020}
}
```
```
@inproceedings{jiang2022joint,
  title={Joint video summarization and moment localization by cross-task sample transfer},
  author={Jiang, Hao and Mu, Yadong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16388--16398},
  year={2022}
}
```
```
@article{ghauri2021MSVA, 
   title={SUPERVISED VIDEO SUMMARIZATION VIA MULTIPLE FEATURE SETS WITH PARALLEL ATTENTION},
   author={Ghauri, Junaid Ahmed and Hakimov, Sherzod and Ewerth, Ralph}, 
   Conference={IEEE International Conference on Multimedia and Expo (ICME)}, 
   year={2021} 
}
```
```
@INPROCEEDINGS{9666088,
    author    = {Apostolidis, Evlampios and Balaouras, Georgios and Mezaris, Vasileios and Patras, Ioannis},
    title     = {Combining Global and Local Attention with Positional Encoding for Video Summarization},
    booktitle = {2021 IEEE International Symposium on Multimedia (ISM)},
    month     = {December},
    year      = {2021},
    pages     = {226-234}
}
```
```
@InProceedings{islam2020position,
   title={How much Position Information Do Convolutional Neural Networks Encode?},
   author={Islam, Md Amirul and Jia, Sen and Bruce, Neil},
   booktitle={International Conference on Learning Representations},
   year={2020}
 }
```
```
@article{zhu2022relational,
  title={Relational reasoning over spatial-temporal graphs for video summarization},
  author={Zhu, Wencheng and Han, Yucheng and Lu, Jiwen and Zhou, Jie},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={3017--3031},
  year={2022},
  publisher={IEEE}
}
```
```
@inproceedings{li2023progressive,
  title={Progressive Video Summarization via Multimodal Self-supervised Learning},
  author={Li, Haopeng and Ke, Qiuhong and Gong, Mingming and Drummond, Tom},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5584--5593},
  year={2023}
}
```
```
@article{hsu2023video,
  title={Video summarization with spatiotemporal vision transformer},
  author={Hsu, Tzu-Chun and Liao, Yi-Sheng and Huang, Chun-Rong},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
```
```
@misc{fajtl2018summarizing,
    title={Summarizing Videos with Attention},
    author={Jiri Fajtl and Hajar Sadeghi Sokeh and Vasileios Argyriou and Dorothy Monekosso and Paolo Remagnino},
    year={2018},
    eprint={1812.01969},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
```
@article{li2022video,
  title={Video Joint Modelling Based on Hierarchical Transformer for Co-summarization},
  author={Li, Haopeng and Ke, Qiuhong and Gong, Mingming and Zhang, Rui},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```

[Back to top](https://github.com/thswodnjs3/CSTA?tab=readme-ov-file#csta-cnn-based-spatiotemporal-attention-for-video-summarization-cvpr-2024-paper)↑
