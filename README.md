# The SOTA model for street view classification
This a project for city building sence classification. This tool is created for the multilabel building classification at a published paper:

Zhang, J., Fukuda, T., & Yabuki, N. (2021). Development of a City-Scale Approach for Façade Color Measurement with Building Functional Classification Using Deep Learning and Street View Images. ISPRS International Journal of Geo-Information, 10(8), 551.

![image](https://user-images.githubusercontent.com/68632919/151101377-46e8bc39-17e0-4058-860b-b90b97e20e2e.png)

# Useage

install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```

install prefetch_generator
```
pip install prefetch_generator
```

install imgaug
```
pip install git+https://github.com/aleju/imgaug
```

# Pre-Model
using efficientnet_b2 with multi-label 
https://drive.google.com/drive/folders/1Q-cbezLxOx8DcsPG1EnEXiAdYeMDxoHy?usp=sharing
