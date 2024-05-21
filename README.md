# CSTA: CNN-based Spatiotemporal Attention for Video Summarization (CVPR 2024 paper)
The official code of "CSTA: CNN-based Spatiotemporal Attention for Video Summarization"

# Additional codes and explanations are continually updated...

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
