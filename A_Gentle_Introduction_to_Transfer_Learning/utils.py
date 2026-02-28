import os
import sys
import glob
import json
import time
import copy
import random
import shutil
import tarfile
import subprocess
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from urllib.request import urlretrieve
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision import models
from torch.optim import lr_scheduler, SGD
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d


def get_number_processors():
    """Get the number of processors in a CPU.
    
    Returns:
        num (int): Number of processors.
    
    Examples:
        >>> get_number_processors()
        4
    """
    try:
        num = os.cpu_count()
    except Exception:
        import multiprocessing #force exception in case mutiprocessing is not installed
        num = multiprocessing.cpu_count()
    return num


def get_gpu_name():
    """Get the GPU names in the system.

    Returns:
        gpu_names (list): List of GPU name strings.

    Examples:
        >>> get_gpu_name()
        ['Tesla M60', 'Tesla M60', 'Tesla M60', 'Tesla M60']
    """
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)


def get_gpu_memory():
    """Get the memory of the GPUs in the system.

    Returns:
        gpu_memory (list): List of GPU memory strings.

    Examples:
        >>> get_gpu_memory()
        ['8123 MiB', '8123 MiB', '8123 MiB', '8123 MiB']
    """
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").replace('\r','').split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)

        
def get_cuda_version():
    """Get the CUDA version.

    Returns:
        version (str): CUDA version string or a message if CUDA is not found.

    Examples:
        >>> get_cuda_version()
        'CUDA Version 8.0.61'
    """
    if sys.platform == 'win32':
        raise NotImplementedError("Implement this!")
    elif sys.platform == 'linux':
        path = '/usr/local/cuda/version.txt'
        if os.path.isfile(path):
            with open(path, 'r') as f:
                data = f.read().replace('\n','')
            return data
        else:
            return "No CUDA in this machine"
    elif sys.platform == 'darwin':
        raise NotImplementedError("Find a Mac with GPU and implement this!")
    else:
        raise ValueError("Not in Windows, Linux or Mac")
        
    
def format_dictionary(dct, indent=4):
    """Format a dictionary to be printed.

    Args:
        dct (dict): Dictionary.
        indent (int): Indentation value.

    Returns:
        result (str): Formatted dictionary ready to be printed.

    Examples:
        >>> dct = {'bkey':1, 'akey':2}
        >>> print(format_dictionary(dct))
        {
            "akey": 2,
            "bkey": 1
        }
    """
    return json.dumps(dct, indent=indent, sort_keys=True)



def get_filenames_in_folder(folderpath):
    """Return the file names in a folder.

    Args:
        folderpath (str): Folder path.

    Returns:
        filenames (list): Sorted list of file names.

    Examples:
        >>> get_filenames_in_folder('C:/run3x/codebase/python/minsc')
        ['paths.py', 'system_info.py', '__init__.py']
    """
    names = [os.path.basename(x) for x in glob.glob(os.path.join(folderpath, '*')) if os.path.isfile(x)]
    return sorted(names)


def get_files_in_folder_recursively(folderpath):
    """Return the files inside a folder recursively.

    Args:
        folderpath (str): Folder path.

    Returns:
        filelist (list): Sorted list of relative file paths.

    Examples:
        >>> get_files_in_folder_recursively(r'C:\\run3x\\codebase\\command_line')
        ['linux\\compress.txt', 'linux\\paths.txt', 'windows\\resources_management.txt']
    """
    if folderpath[-1] != os.path.sep: #Add final '/' if it doesn't exist
        folderpath += os.path.sep
    names = [x.replace(folderpath,'') for x in glob.iglob(folderpath+'/**', recursive=True) if os.path.isfile(x)]
    return sorted(names)



def _create_sets_folders(root_folder, sets_names, target_folder):
    """Create the train/val folder structure for a given class."""
    for s in sets_names:
        dest = os.path.join(root_folder, s, target_folder)
        os.makedirs(dest, exist_ok=True)
          
                
def split_list(py_list, perc_size=[0.8, 0.2], shuffle=False):
    """Split a list into weighted chunks.

    Args:
        py_list (list): A list of elements.
        perc_size (list): The percentage size of each chunk. Must sum to 1.
        shuffle (bool): Whether to shuffle the list before splitting.

    Returns:
        result_list (list of list): A list of lists with the chunks.

    Examples:
        >>> split_list(list(range(7)),[0.47,0.33,0.2])
        [[0, 1, 2], [3, 4, 5], [6]]
        >>> split_list(list(range(10)),[0.6,0.4], True)
        [[1, 2, 3, 6, 9, 5], [4, 8, 0, 7]]
    """
    assert sum(perc_size) == 1, "Percentage sizes do not sum to 1"
    l = py_list[:]
    if shuffle:
        random.shuffle(l)
    # Turn percentages into values between 0 and 1
    splits = np.cumsum(perc_size)

    # Split doesn't need last percent, it will just take what is left
    splits = splits[:-1]

    # Turn values into indices
    splits *= len(l)

    # Turn double indices into integers.
    splits = splits.round().astype(int)

    return [list(chunks) for chunks in np.split(l, splits)]


def split_dataset_folder(root_folder, dest_folder, sets_names=['train','val'], sets_sizes=[0.8,0.2], shuffle=False, verbose=False):
    """Split an image dataset folder into PyTorch ImageFolder format.

    Transforms a flat class-based structure into train/val splits::

        Before:                After:
        class1/                train/
          img1.jpg               class1/
          img2.jpg                 img1.jpg
        class2/                  class2/
          img1.jpg                 img1.jpg
          img2.jpg             val/
                                 class1/
                                   img2.jpg
                                 class2/
                                   img2.jpg

    Args:
        root_folder (str): Path to the source dataset with one folder per class.
        dest_folder (str): Path where the split dataset will be created.
        sets_names (list): Names of the splits (e.g. ['train', 'val']).
        sets_sizes (list): Proportion of each split. Must sum to 1.
        shuffle (bool): Whether to shuffle files before splitting.
        verbose (bool): Print progress information.
    """
    assert sum(sets_sizes) == 1, "Data set sizes do not sum to 1"
    # List class directories (not files) in root_folder
    class_folders = sorted(d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d)))
    for folder in class_folders:
        if verbose: print("Folder: ", folder)
        _create_sets_folders(dest_folder, sets_names, folder)
        files = get_filenames_in_folder(os.path.join(root_folder, folder))
        files_split = split_list(files, sets_sizes, shuffle)
        for split, set_name in zip(files_split, sets_names):
            for f in split:
                orig = os.path.join(root_folder, folder, f)
                dest = os.path.join(dest_folder, set_name, folder)
                if verbose: print("Copying {} into {}".format(orig, dest))
                shutil.copy2(orig, dest)

                
def validate_image_dataset(root_folder, remove=False, verbose=False):
    """Validate all images in a dataset folder, reporting or removing corrupt files.

    Args:
        root_folder (str): Path to the dataset.
        remove (bool): If True, delete corrupt files. If False, only report them.
        verbose (bool): Print progress information.

    Returns:
        list: Paths of corrupt or truncated files found.
    """
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    bad_files = []
    files = get_files_in_folder_recursively(root_folder)
    for f in files:
        if os.path.splitext(f)[1].lower() not in IMAGE_EXTENSIONS:
            continue
        filepath = os.path.join(root_folder, f)
        try:
            img = Image.open(filepath)
            img.verify()
        except Exception as e:
            if verbose:
                print("Corrupt image {}: {}".format(filepath, e))
            bad_files.append(filepath)
            if remove:
                os.remove(filepath)
                if verbose:
                    print("Removed {}".format(filepath))
    if verbose:
        print("Validation complete: {}/{} bad files found.".format(len(bad_files), len(files)))
    return bad_files


def convert_image_dataset_to_grayscale(root_folder, dest_folder, verbose=False):
    """Convert all images in a dataset folder to grayscale.

    Args:
        root_folder (str): Path to the source dataset.
        dest_folder (str): Path where grayscale images will be saved.
        verbose (bool): Print progress information.
    """
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    files = get_files_in_folder_recursively(root_folder)
    for f in files:
        if os.path.splitext(f)[1].lower() not in IMAGE_EXTENSIONS:
            if verbose: print("Skipping non-image file: {}".format(f))
            continue
        filename = os.path.join(root_folder, f)
        if verbose: print("Converting {} to grayscale".format(filename))
        try:
            img = Image.open(filename)
            img_gray = img.convert('L')
        except Exception as e:
            if verbose: print("Skipping {}: {}".format(filename, e))
            continue
        dest = os.path.join(dest_folder, f)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        img_gray.save(dest)
            
            
def create_dataset(data_dir, batch_size=32, sets=['train', 'val'], verbose=False):
    """Create PyTorch DataLoaders from an ImageFolder dataset.

    Expects ``data_dir`` to contain subdirectories named after each set (e.g.
    ``train/``, ``val/``), each with one folder per class.

    Args:
        data_dir (str): Root path of the dataset.
        batch_size (int): Batch size for the DataLoaders.
        sets (list): Names of the dataset splits to load.
        verbose (bool): Print dataset statistics.

    Returns:
        dataloaders (dict): Dictionary mapping set names to DataLoader objects.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in sets}
    num_workers = min(get_number_processors(), 4)
    use_cuda = torch.cuda.is_available()
    dataloaders = {x: DataLoader(image_datasets[x],
                                 batch_size=batch_size,
                                 shuffle=(x == 'train'),
                                 num_workers=num_workers,
                                 pin_memory=use_cuda,
                                 persistent_workers=False)
                   for x in sets}

    if verbose:
        dataset_sizes = {x: len(image_datasets[x]) for x in sets}
        class_names = dataloaders[sets[0]].dataset.class_to_idx
        print("There are {} clases in the dataset: {}".format(len(class_names), format_dictionary(class_names)))
        print("Sets sizes: ", format_dictionary(dataset_sizes))
        for x in sets:   
            c = Counter(item[1] for item in image_datasets[x])
            c = dict(c)
            print("Number of items in set {}: {}".format(x, c))
    return dataloaders


def plot_pytorch_data_stream(dataobject, max_images=8, title=True):
    """Plot a batch of images from a PyTorch DataLoader.

    Args:
        dataobject (DataLoader): A PyTorch DataLoader.
        max_images (int): Maximum number of images to display.
        title (bool): Whether to show class names as the plot title.
    """
    inputs, classes = next(iter(dataobject))  
    if max_images > dataobject.batch_size:
        max_images = dataobject.batch_size
        print("Plotting only {} images, which is the batch size".format(max_images))
    inputs = inputs[:max_images,:,:,:]
    classes = classes[:max_images]
    out = torchvision.utils.make_grid(inputs)
    inp = out.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.axis('off')
    if title:
        class_names = dataobject.dataset.classes
        names = [class_names[x] for x in classes]
        plt.title(names)

        
def finetune(dataloaders, model_name, sets, num_epochs, num_gpus, lr, momentum, lr_step, lr_epochs, verbose=False):
    """Finetune all layers of a pretrained model on a new dataset.

    Args:
        dataloaders (dict): Dictionary of DataLoaders keyed by set name.
        model_name (str): Name of the torchvision model (e.g. 'resnet18').
        sets (list): Dataset split names (e.g. ['train', 'val']).
        num_epochs (int): Number of training epochs.
        num_gpus (int): Number of GPUs to use.
        lr (float): Learning rate.
        momentum (float): SGD momentum.
        lr_step (float): Factor to decay the learning rate by.
        lr_epochs (int): Number of epochs between each LR decay step.
        verbose (bool): Print training progress.

    Returns:
        model (nn.Module): The finetuned model with the best validation weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Class adaptation
    num_class = len(dataloaders[sets[0]].dataset.class_to_idx)
    model_ft = models.__dict__[model_name](weights="DEFAULT")
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_class)

    #gpus
    if num_gpus > 1 and torch.cuda.device_count() > 1:
        model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)


    #loss
    criterion = nn.CrossEntropyLoss()

    # All parameters are being optimized
    optimizer = SGD(model_ft.parameters(), lr=lr, momentum=momentum)

    # Decay LR by a factor of lr_step every lr_epochs epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_epochs, gamma=lr_step)
    model_ft = train_model(dataloaders, model_ft, sets, criterion, optimizer, exp_lr_scheduler,
                           num_epochs=num_epochs, verbose=verbose)
    return model_ft


def freeze_and_train(dataloaders, model_name, sets, num_epochs, num_gpus, lr, momentum, lr_step, lr_epochs, verbose=False):
    """Freeze all layers except the last one and train only the final classification layer.

    Args:
        dataloaders (dict): Dictionary of DataLoaders keyed by set name.
        model_name (str): Name of the torchvision model (e.g. 'resnet18').
        sets (list): Dataset split names (e.g. ['train', 'val']).
        num_epochs (int): Number of training epochs.
        num_gpus (int): Number of GPUs to use.
        lr (float): Learning rate.
        momentum (float): SGD momentum.
        lr_step (float): Factor to decay the learning rate by.
        lr_epochs (int): Number of epochs between each LR decay step.
        verbose (bool): Print training progress.

    Returns:
        model (nn.Module): The trained model with the best validation weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Class adaptation
    num_class = len(dataloaders[sets[0]].dataset.class_to_idx)
    model_conv = models.__dict__[model_name](weights="DEFAULT")
    for param in model_conv.parameters(): #params have requires_grad=True by default
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_class)

    #gpus
    if num_gpus > 1 and torch.cuda.device_count() > 1:
        model_conv = nn.DataParallel(model_conv)
    model_conv = model_conv.to(device)


    #loss
    criterion = nn.CrossEntropyLoss()

    # Only parameters of final layer are being optimized
    if num_gpus > 1 and torch.cuda.device_count() > 1:
        params = model_conv.module.fc.parameters()
    else:
        params = model_conv.fc.parameters()
    optimizer = SGD(params, lr=lr, momentum=momentum)

    # Decay LR by a factor of lr_step every lr_epochs epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_epochs, gamma=lr_step)
    model_conv = train_model(dataloaders, model_conv, sets, criterion, optimizer, exp_lr_scheduler,
                             num_epochs=num_epochs, verbose=verbose)
    return model_conv


def train_model(dataloaders, model, sets, criterion, optimizer, scheduler, num_epochs=25, verbose=False):
    """Train a PyTorch model and track metrics per epoch.

    Args:
        dataloaders (dict): Dictionary of DataLoaders keyed by set name.
        model (nn.Module): The model to train.
        sets (list): Dataset split names (e.g. ['train', 'val']).
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        num_epochs (int): Number of training epochs.
        verbose (bool): Print training progress.

    Returns:
        model (nn.Module): The model with the best validation weights loaded.
        metrics (dict): Dictionary with keys 'train_acc', 'val_acc', 'train_loss',
            'val_loss', and 'cm' (confusion matrices), each a list per epoch.
    """
    device = next(model.parameters()).device
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler(enabled=use_amp)
    since = time.time()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in sets}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_classes = len(dataloaders[sets[0]].dataset.classes)
    metrics = {'train_acc':[],'val_acc':[],'train_loss':[],'val_loss':[], 'cm':[]}
    for epoch in range(num_epochs):
        if verbose:
            print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in sets:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                # forward with automatic mixed precision
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                if phase == 'val':
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            if phase == 'train':
                scheduler.step()

            #metrics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            if verbose: print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'train':
                metrics['train_acc'].append(epoch_acc)
                metrics['train_loss'].append(epoch_loss)
            else:
                metrics['val_acc'].append(epoch_acc)
                metrics['val_loss'].append(epoch_loss)
                # Build confusion matrix
                all_preds = torch.cat(all_preds).numpy()
                all_labels = torch.cat(all_labels).numpy()
                cm = np.zeros((num_classes, num_classes), dtype=int)
                for pred, label in zip(all_preds, all_labels):
                    cm[label, pred] += 1
                metrics['cm'].append(cm)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics


def available_models():
    """Return available torchvision model names, callable via ``models.__dict__[name]``.

    Returns:
        model_names (list): Sorted list of model name strings.
    """
    model_names = sorted(name for name in models.__dict__  if name.islower() and not name.startswith("__") and 
                         callable(models.__dict__[name]))
    return model_names


def plot_metrics(metrics, title=None):
    """Plot training and validation accuracy and loss curves.

    Args:
        metrics (dict): Dictionary with keys 'train_acc', 'val_acc', 'train_loss'
            and 'val_loss', each containing a list of values per epoch.
        title (str or None): Optional title for the plot.
    """
    max_epochs = len(metrics['train_acc']) + 1
    epochs = range(1, max_epochs)
    epochs_dx = np.linspace(epochs[0], epochs[-1], num=max_epochs*4, endpoint=True)
    s_train_acc = interp1d(epochs, metrics['train_acc'], kind='cubic')    
    s_val_acc = interp1d(epochs, metrics['val_acc'], kind='cubic')    
    s_train_loss = interp1d(epochs, metrics['train_loss'], kind='cubic')    
    s_val_loss = interp1d(epochs, metrics['val_loss'], kind='cubic')    

    fig, ax = plt.subplots(1, 2)
    fig.subplots_adjust(right=2, top=0.85)
    if title is not None:
        st = fig.suptitle(title, fontsize=16)
        st.set_x(1)
    ax[0].plot(epochs, metrics['train_acc'], 'b.', label='train')
    ax[0].plot(epochs_dx, s_train_acc(epochs_dx), 'b')
    ax[0].plot(epochs, metrics['val_acc'], 'g.', label='val')
    ax[0].plot(epochs_dx, s_val_acc(epochs_dx), 'g')
    ax[0].legend( loc="lower right")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epochs")
    ax[0].xaxis.set_major_locator(MultipleLocator(1))#only integers in axis multiples of 1
    
    ax[1].plot(epochs, metrics['train_loss'], 'b.', label='train')
    ax[1].plot(epochs_dx, s_train_loss(epochs_dx), 'b')
    ax[1].plot(epochs, metrics['val_loss'], 'g.', label='val')
    ax[1].plot(epochs_dx, s_val_loss(epochs_dx), 'g')
    ax[1].legend(loc="upper right")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].xaxis.set_major_locator(MultipleLocator(1))
    plt.show()
    
    
def download_hymenoptera(out_dir):
    """Download the Hymenoptera dataset (ants vs bees) from the PyTorch tutorial.

    Args:
        out_dir (str): Directory where the dataset will be extracted.

    Returns:
        dest (str): Path to the extracted dataset folder.
    """
    import zipfile
    url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
    dest = os.path.join(out_dir, 'hymenoptera_data')
    if os.path.isdir(dest) and os.listdir(dest):
        print("Dataset already downloaded in {}".format(dest))
        return dest
    os.makedirs(out_dir, exist_ok=True)
    print("Downloading {}".format(url))
    filepath = os.path.join(out_dir, 'hymenoptera_data.zip')
    urlretrieve(url, filepath)
    print("Extracting files...")
    with zipfile.ZipFile(filepath, 'r') as z:
        z.extractall(out_dir)
    os.remove(filepath)
    print("Dataset ready at {}".format(dest))
    return dest


def download_caltech256(out_dir):
    """Download the Caltech 256 dataset from the official Caltech data repository.

    Args:
        out_dir (str): Directory where the dataset will be extracted.

    Returns:
        dest (str): Path to the extracted dataset folder.
    """
    url = 'https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar'
    dest = os.path.join(out_dir, '256_ObjectCategories')
    if os.path.isdir(dest) and os.listdir(dest):
        print(f"Dataset already downloaded in {dest}")
        return dest
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading Caltech 256 from {url}")
    filepath = os.path.join(out_dir, '256_ObjectCategories.tar')
    urlretrieve(url, filepath)
    print("Extracting files...")
    with tarfile.open(filepath) as tar:
        tar.extractall(path=out_dir, filter='data')
    os.remove(filepath)
    print(f"Dataset ready at {dest}")
    return dest


def _load_kaggle_env():
    """Load Kaggle credentials from .env file if not already set."""
    if os.environ.get('KAGGLE_KEY'):
        return
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if value:
                        os.environ[key] = value


def download_simpsons(out_dir):
    """Download the Simpsons Characters dataset from Kaggle.

    Requires kagglehub and a valid Kaggle API key (via .env or ~/.kaggle/kaggle.json).

    Args:
        out_dir (str): Directory where the dataset will be saved.

    Returns:
        dest (str): Path to the dataset folder.
    """
    _load_kaggle_env()
    import kagglehub
    dest = os.path.join(out_dir, 'simpsons')
    if os.path.isdir(dest) and os.listdir(dest):
        print(f"Dataset already downloaded in {dest}")
        return dest
    os.makedirs(out_dir, exist_ok=True)
    print("Downloading Simpsons dataset from Kaggle...")
    path = kagglehub.dataset_download("alexattia/the-simpsons-characters-dataset")
    print(f"Downloaded to cache: {path}")
    # Copy to our data directory
    shutil.copytree(path, dest, dirs_exist_ok=True)
    print(f"Dataset ready at {dest}")
    return dest


def download_dogs_vs_cats(out_dir):
    """Download the Dogs vs Cats dataset from Kaggle.

    Requires kagglehub and a valid Kaggle API key (via .env or ~/.kaggle/kaggle.json).

    Args:
        out_dir (str): Directory where the dataset will be saved.

    Returns:
        dest (str): Path to the dataset folder.
    """
    _load_kaggle_env()
    import kagglehub
    dest = os.path.join(out_dir, 'dogs_vs_cats')
    if os.path.isdir(dest) and os.listdir(dest):
        print(f"Dataset already downloaded in {dest}")
        return dest
    os.makedirs(out_dir, exist_ok=True)
    print("Downloading Dogs vs Cats dataset from Kaggle...")
    path = kagglehub.dataset_download("karakaggle/kaggle-cat-vs-dog-dataset")
    print(f"Downloaded to cache: {path}")
    shutil.copytree(path, dest, dirs_exist_ok=True)
    print(f"Dataset ready at {dest}")
    return dest

