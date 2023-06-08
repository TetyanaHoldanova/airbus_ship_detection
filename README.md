# airbus_ship_detection
## Project Description
Deep Learning models for detecting ships in satellite images [Kaggle Airbus Competition](https://www.kaggle.com/c/airbus-ship-detection). 

## Dataset
The dataset used consisted of a set of images for training and testing, as well as a .csv file that contained rle-encoded strings for the training masks.
The dataset was imbalanced, so the following was performed during image preprocessing:
* Delete damaged photos with size < 50 kb
* Resize images from 768x768 to 256x256 (optional step)
* Stratified sampling (2000 pictures for each number of depicted ships)

## Project structure
```
├── .gitignore
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.        
├── README.md 
├── setup.py           <- Make this project pip installable with `pip install -e`
├── notebooks
│   └── exploratory_data_analysis.ipynb
├── src
│   ├── image_segmentation
│   │   ├── assets
│   │   │   ├── data
│   │   │   │   ├── test_v2
│   │   │   │   ├── train_v2
│   │   │   │   ├── results.csv
│   │   │   │   ├── train_ship_segmentations_v2.csv
│   │   │   ├── model
│   │   │   │   ├── model.h5
│   │   │   │   ├── model_weight.h5
│   │   ├── model
│   │   │   ├── unet.py
│   │   ├── utils
│   │   │   ├── dataset.py
│   │   │   ├── generators.py
│   │   │   ├── metrics.py
│   │   │   ├── rle.py
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── train.py
│   │   ├── test.py
```

## Installation
## Usage
## Results
