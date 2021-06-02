# DP-ATT

DP-ATT combines [DeepPrivacy](https://github.com/hukkelas/DeepPrivacy)  with [AttGAN](https://github.com/elvisyjlin/AttGAN-PyTorch) to preserve gender and age in de-identfied CCTV footage. It also includes a model for changing skin tone to dark(light.


## Installation
Install the following: 
- Pytorch  >= 1.7.0
- Torchvision >= 0.6.0
- NVIDIA Apex (If you want to train any DeepPrivacy models - not needed for inference)
- Python >= 3.6

Simply by running our `setup.py` file:

```
python3 setup.py install
```
or with pip:
```
pip install git+https://github.com/skoskjei/DP-ATT/
```

#### Docker

Docker image can be built by running:
```bash
cd docker/

docker build -t deep_privacy . 
```
## Models

For gender and age estimation we have used [Age and Gender Estimation](https://github.com/yu4u/age-gender-estimation). 
Download the pre-trained weights into the folder age_and_gender_estimation.

For AttGAN it is possible to use two separate models, one with only gender and age ([DP-ATT](https://drive.google.com/file/d/12EgVJlQ-btiMRWkPdqXHbAdSf9Tr8bFl/view)) and one with gender, age and skin tone ([DP-ATT-S](https://drive.google.com/file/d/1kVwggjaS6FdOg8hBgMkg9m7hbz86UwrW/view)). 
The trained models should be placed in a folder named 'output', inside the folder 'attgan'. Instead of using L1 for reconstruction loss, we use [MS-SSIM_L1_LOSS](https://github.com/psyrocloud/MS-SSIM_L1_LOSS).

## Usage
```
python3 anonymize.py -s input_image.png -t output_path.png --experiment_name experiment_name --checkpoint checkpoint
```
You can change the model with the "-m" or "--model" flag [see model zoo](https://github.com/hukkelas/DeepPrivacy).
The cli accepts image files, video files, and directories.

The cli is also available outside the folder `python -m deep_privacy.cli`.

Also check out `python -m deep_privacy.cli -h ` for more arguments.

## License 

All code is under MIT license, except the following. 

Code under [deep_privacy/detection](deep_privacy/detection):
- DSFD is taken from [https://github.com/hukkelas/DSFD-Pytorch-Inference](https://github.com/hukkelas/DSFD-Pytorch-Inference) and follows APACHE-2.0 License
- Mask R-CNN implementation is taken from Pytorch source code at [pytorch.org](https://pytorch.org/docs/master/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
- FID calculation code is taken from the official tensorflow implementation: [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR)

The IMDB-WIKI dataset used in [Age and Gender Estimation](https://github.com/yu4u/age-gender-estimation) is originally provided under the [following conditions](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).




