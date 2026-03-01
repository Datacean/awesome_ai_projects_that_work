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
python finetune.py -d ./data/hymenoptera_data -b 256 -m resnet18 -lr 0.002 -e 15 -f 1

# Freeze and train
python finetune.py -d ./data/hymenoptera_data -b 256 -m resnet18 -lr 0.002 -e 15 -f 0
```

## Kaggle Setup (for Simpsons and Dogs vs Cats datasets)

1. Copy `.env.template` to `.env` and fill in your Kaggle credentials:

```bash
cp .env.template .env
# Edit .env with your KAGGLE_USERNAME and KAGGLE_KEY
```

You can find your API key at https://www.kaggle.com/settings (Create New Token).

## Datasets Used

All datasets can be downloaded automatically from the notebook (`data_prep.ipynb`):

| Dataset | Classes | Images | Source | Download |
|---------|:-------:|:------:|--------|----------|
| Hymenoptera | 2 | 397 | [PyTorch tutorial](https://download.pytorch.org/tutorial/hymenoptera_data.zip) | Automatic |
| Simpsons | 20 | 19,548 | [Kaggle](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset) | Requires Kaggle API key |
| Dogs vs Cats | 2 | 25,000 | [Kaggle](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset) | Requires Kaggle API key |
| Caltech 256 | 257 | 30,607 | [Caltech](https://data.caltech.edu/records/nyy15-4j048) | Automatic |

## Results (ResNet18, 15 epochs, batch size 256, LR 0.002)

| Dataset | Finetuning Accuracy | Freeze Accuracy |
|---------|:-------------------:|:---------------:|
| Hymenoptera | 94.8% | 86.9% |
| Hymenoptera gray | 84.3% | 64.1% |
| Simpsons | 95.4% | 69.3% |
| Simpsons gray | 93.3% | 60.3% |
| Dogs vs Cats | 99.2% | 98.4% |
| Dogs vs Cats gray | 99.0% | 98.0% |
| Caltech 256 | 76.6% | 70.0% |
| Caltech 256 gray | 72.5% | 59.3% |

## Project Files

- `Intro_Transfer_Learning.ipynb` — Main notebook with explanations and experiments
- `finetune.py` — CLI script for training
- `utils.py` — Dataset loading, training loop, visualization utilities
- `data_prep.ipynb` — Dataset splitting and grayscale conversion
- `convert_image_dataset_to_grayscale.py` — CLI tool for grayscale conversion
