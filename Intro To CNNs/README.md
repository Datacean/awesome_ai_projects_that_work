# Introduction to Convolutional Neural Networks (CNN) — example summary

This folder contains a short Jupyter notebook that demonstrates two related concepts:

- Basic image convolutions and filters applied to the classic "Lenna" image using OpenCV (sharpening, Laplacian, Sobel X/Y).
- Training a small Convolutional Neural Network (LeNet) on the MNIST handwritten digit dataset using PyTorch.

What the notebook does (high level):

- Loads and displays the `Lenna.png` image, and applies several 3x3 convolution kernels with `cv2.filter2D` to illustrate how convolution kernels produce sharpening and edge detection effects.
- Downloads the MNIST dataset via `sklearn.datasets.fetch_openml`, normalizes and reshapes the data into PyTorch tensors, and creates DataLoaders for training and testing.
- Implements a small LeNet-style CNN with two convolutional layers and two fully-connected layers in PyTorch, moves the model to GPU if available, and trains for multiple epochs using SGD + CrossEntropyLoss.
- Prints training progress and test set accuracy after each epoch.

Quick notes:

- The notebook is meant for educational/demo purposes — it's small and intentionally simple so you can read and understand each step.
- If you want to run training on a GPU, install a CUDA-compatible PyTorch wheel for your system (example in the notebook shows an index URL for the CUDA 11.6 wheel). Otherwise, CPU-only PyTorch will work but will be slower.

How to run

1. Create a virtual environment (optional but recommended).
2. Install the requirements listed in `requirements.txt`.
3. Open `Intro_CNN_PyTorch.ipynb` with Jupyter Notebook or JupyterLab and run the cells.

Example install & run (replace with your environment commands):

    conda create -n datacean python=3.12 -y
	pip install -r requirements.txt
    conda activate datacean
	jupyter notebook "Intro_CNN_PyTorch.ipynb"

If you plan to use CUDA-enabled PyTorch, follow the official PyTorch install instructions for the correct CUDA version on your machine (the notebook contains an example `pip` command using the PyTorch download index URL).
