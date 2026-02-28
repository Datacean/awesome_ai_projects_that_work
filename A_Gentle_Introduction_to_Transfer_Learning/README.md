# A Gentle Introduction to Transfer Learning

Transfer Learning allows you to reuse an already trained CNN on a new dataset instead of training from scratch. This project demonstrates two strategies:

- **Finetuning**: Train all layers of a pretrained network on the target dataset.
- **Freeze and train**: Freeze all layers except the last one, and only train the final classification layer.

## What You'll Learn

- When to use finetuning vs freeze-and-train
- How dataset size and domain similarity affect transfer learning performance
- Effect of grayscale vs color images on transferability
- Practical training with ResNet18 pretrained on ImageNet

## Quick Start

```bash
uv venv datacean --python 3.12
source datacean/bin/activate
uv pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook Intro_Transfer_Learning.ipynb
```

### Run from the command line

```bash
# Download the Hymenoptera dataset (ants vs bees, 397 images)
python -c "from utils import download_hymenoptera; download_hymenoptera('./data')"

# Finetuning
python finetune.py -d ./data/hymenoptera_data -b 32 -m resnet18 -lr 0.001 -e 15 -f 1

# Freeze and train
python finetune.py -d ./data/hymenoptera_data -b 32 -m resnet18 -lr 0.001 -e 15 -f 0
```

## Datasets Used

| Dataset | Classes | Images | Source |
|---------|:-------:|:------:|--------|
| Hymenoptera | 2 | 397 | [PyTorch tutorial](https://download.pytorch.org/tutorial/hymenoptera_data.zip) |
| Simpsons | 20 | 19,548 | [Kaggle](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset) |
| Dogs vs Cats | 2 | 25,000 | [Kaggle](https://www.kaggle.com/c/dogs-vs-cats) |
| Caltech 256 | 257 | 30,607 | [Caltech](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) |

The Hymenoptera dataset downloads automatically. For the others, download from the links above and use `data_prep.ipynb` to split into train/val folders.

## Results (ResNet18, 15 epochs)

| Dataset | Finetuning Accuracy | Freeze Accuracy |
|---------|:-------------------:|:---------------:|
| Hymenoptera | 95.4% | 96.1% |
| Hymenoptera gray | 92.2% | 90.2% |
| Simpsons | 92.5% | 64.2% |
| Dogs vs Cats | 98.9% | 98.1% |
| Caltech 256 | 67.4% | 54.3% |

## Project Files

- `Intro_Transfer_Learning.ipynb` — Main notebook with explanations and experiments
- `finetune.py` — CLI script for training
- `utils.py` — Dataset loading, training loop, visualization utilities
- `data_prep.ipynb` — Dataset splitting and grayscale conversion
- `convert_image_dataset_to_grayscale.py` — CLI tool for grayscale conversion
