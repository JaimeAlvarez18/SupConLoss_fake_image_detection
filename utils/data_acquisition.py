"""
Utility functions and Dataset classes for loading images and computing embeddings.

Notes:
- This file is intentionally not changing logic, only adding comments and minor formatting.
- Keep behavior identical to the original implementation.
"""
import numpy as np
from glob import glob
import random
random.seed(42)
import os
from sklearn.metrics import auc
from torch.utils.data import Dataset
import cv2
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch.amp as amp

import gc
import os


class test_dataset(Dataset):
    """PyTorch Dataset wrapper that loads images from file paths and returns (tensor_image, label).

    Expected input:
    - X: list of file paths
    - y: list/array of labels (one per file path)
    - device: device string (not used for storage, only kept for API compatibility)
    - size: integer resolution to which images are resized (size x size)
    """
    def __init__(self, X, y, device, size):
        self.device = device
        self.routes = X
        self.labels = y
        self.size = size

    def __getitem__(self, index):
        im1 = self.routes[index]
        y = self.labels[index]

        # Load image from disk (BGR by default with OpenCV)
        image1 = cv2.imread(im1)

        # If image is corrupted or missing, warn and remove the file
        if type(image1) == type(None):
            print(f"Warning: Image at {im1}")
            os.remove(im1)

        # Scale to [0, 1]
        image1 = image1 / 255.0

        # Resize to the target resolution
        image1 = cv2.resize(image1, (self.size, self.size))

        # Convert HWC to CHW
        image1 = np.transpose(image1, [2, 0, 1])

        # Convert to torch tensor
        image1 = torch.from_numpy(image1).to(torch.float32)
        y = torch.tensor(y).to(torch.float32)

        return image1, y

    def __len__(self):
        return len(self.routes)


class BIG_DATALOADER(Dataset):
    """Dataset that returns raw file paths and labels (used when loading outside the normal pipeline)."""
    def __init__(self, X, y, device, size):
        self.device = device
        self.routes = X
        self.labels = y
        self.size = size

    def __getitem__(self, index):
        im1 = self.routes[index]
        y = self.labels[index]
        return im1, y

    def __len__(self):
        return len(self.routes)


def read_files(routes):
    """Read a list of image file paths into a single torch tensor (float16).

    This helper loads images, rescales to 256x256 and returns a tensor of shape (N, C, H, W).
    """
    all_images = []
    for route in routes:
        image1 = cv2.imread(route)

        # If corrupted, warn and remove
        if type(image1) == type(None):
            print(f"Warning: Image at {route}")
            os.remove(route)

        # Scale and resize
        image1 = image1 / 255.0
        image1 = cv2.resize(image1, (256, 256))

        # Convert HWC -> CHW
        image1 = np.transpose(image1, [2, 0, 1])
        all_images.append(image1)

    # Convert to tensor (float16) and return
    all_images = torch.from_numpy(np.array(all_images)).to(torch.float16)
    return all_images


class data_set_with_nature():
    """Load dataset splits with a dedicated 'nature' (real) class included.

    The dataset folder structure is expected to include train/val subfolders
    and each dataset contains ai/nature image subfolders.
    """
    def __init__(self, route):
        self.route = route
        self.route_nature = "Datasets/GenImage_sampled/ADM/"

    def get_data(self, split, get_all=False, train_size=0.9, random_state=5):
        """
        Return train/val/test splits and their labels.

        If get_all is True, labels for test are encoded as integers via LabelEncoder.
        The function also prints the encoder classes and training distribution.
        """
        train = []
        test = []
        y_train = []
        y_test = []

        # Define split-specific folds and indices
        if split == "ES1":
            vals = ['stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'wukong', "glide", "Midjourney"]
            test_indices = [2, 3, 5, 6, 8]
            train_indices = [0, 1, 4, 7]
        elif split == "ES2":
            vals = ['ADM', 'BigGan', 'Midjourney', "stable_diffusion_v_1_5", "vqdm"]
            test_indices = [0, 1, 2, 6, 7]
            train_indices = [3, 4, 5, 8]
        elif split == "ES3":
            vals = ['ADM', 'BigGan', 'wukong', "glide", "vqdm"]
            test_indices = [0, 1, 3, 7, 8]
            train_indices = [2, 4, 5, 6]
        elif split == "ES4":
            vals = ['stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'wukong', "glide", "BigGan"]
            test_indices = [1, 3, 5, 6, 8]
            train_indices = [0, 2, 4, 7]
        elif split == "ES5":
            vals = []
            test_indices = []
            train_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        elif split == "Split-1":
            vals = ["glide", "ADM", "stable_diffusion_v_1_5", "BigGan"]
            test_indices = [0, 1, 3, 6]
            train_indices = [2, 4, 5, 7, 8]
        elif split == "Split-2":
            vals = ["wukong", "glide", "ADM", "stable_diffusion_v_1_5"]
            test_indices = [0, 2, 6, 8]
            train_indices = [1, 3, 4, 5, 7]
        elif split == "Split-3":
            vals = ["wukong", "glide", "stable_diffusion_v_1_5", "Midjourney"]
            test_indices = [2, 3, 6, 8]
            train_indices = [0, 1, 4, 5, 7]
        elif split == "Split-4":
            vals = ["wukong", "stable_diffusion_v_1_4", "Midjourney", "vqdm"]
            test_indices = [2, 5, 7, 8]
            train_indices = [0, 1, 3, 4, 6]

        # Walk the top-level directory and gather datasets
        itera = os.walk(self.route)
        datasets = next(iter(itera))[1]
        for _, dataset in enumerate(datasets):
            direct = self.route + dataset + '/'
            if get_all:
                # When requested, include both train and val ai files in lists
                test.append(glob(direct + 'val/ai/*.PNG') + glob(direct + 'val/ai/*.png'))
                y_test.append([dataset] * len(test[len(test) - 1]))
                train.append(glob(direct + 'train/ai/*.PNG') + glob(direct + 'train/ai/*.png'))
                y_train.append([dataset] * len(train[len(train) - 1]))
            else:
                if dataset in vals and get_all == False:
                    test.append(glob(direct + 'val/ai/*.PNG') + glob(direct + 'val/ai/*.png'))
                    y_test.append([dataset] * len(test[len(test) - 1]))
                else:
                    train.append(glob(direct + 'train/ai/*.PNG') + glob(direct + 'train/ai/*.png'))
                    y_train.append([dataset] * len(train[len(train) - 1]))

        # Add nature/real images (validation + training) from the designated ADM folder
        nature1 = glob(self.route_nature + "/val/nature/*.JPEG")
        test.append(nature1)
        y_test.append(["real"] * len(nature1))

        nature1 = glob(self.route_nature + "/train/nature/*.JPEG")
        train.append(nature1)
        y_train.append(["real"] * len(nature1))

        # Flatten nested lists
        train = [item for sublist in train for item in sublist]
        test = [item for sublist in test for item in sublist]
        y_train = [item for sublist in y_train for item in sublist]
        y_test = [item for sublist in y_test for item in sublist]

        # Encode labels (fit on training labels)
        encoder = LabelEncoder().fit(y_train)
        y_train = encoder.transform(y_train)
        if get_all:
            y_test = encoder.transform(y_test)

        print("TRAINING DATA:")
        print(encoder.classes_)
        print(np.unique(np.array(y_train), return_counts=True))

        # Stratified split of training into train/val
        train, val, y_train, y_val = train_test_split(
            train, y_train, train_size=train_size, stratify=y_train, random_state=random_state
        )

        return train, val, test, y_train, y_val, y_test, train_indices, test_indices


class data_set_binary_with_nature():
    """Construct binary datasets where class '8' denotes nature/real images."""
    def __init__(self, route):
        self.route = route
        self.route_nature = "Datasets/GenImage_sampled/ADM"

    def get_data(self, train_size=0.9, random_state=5):
        all_train = []
        all_test = []
        all_y_train = []
        all_y_test = []

        datasets = os.listdir(self.route)
        print(datasets)
        for index, dataset in enumerate(datasets):
            test = []
            train = []
            y_train = []
            y_test = []
            direct = self.route + dataset + '/'

            test.extend(glob(direct + 'val/ai/*.PNG') + glob(direct + 'val/ai/*.png'))
            y_test.extend([index] * len(test))
            lista = glob(direct + 'val/nature/*.JPEG')
            test.extend(lista)
            y_test.extend([8] * len(lista))

            train.extend(glob(direct + 'train/ai/*.PNG') + glob(direct + 'train/ai/*.png'))
            y_train.extend([index] * len(train))

            nature1 = glob(self.route_nature + "/train/nature/*.JPEG")
            train.extend(nature1)
            y_train.extend([8] * len(nature1))

            all_train.append(train)
            all_test.append(test)
            all_y_train.append(y_train)
            all_y_test.append(y_test)

        return all_train, all_test, all_y_train, all_y_test


class data_set_binary_synth():
    """Build a binary synthetic vs real dataset reading files recursively."""
    def __init__(self, route):
        self.route = route

    def get_data(self):
        all_y_train = []
        source_root = self.route + "/train/"
        all_files_train = []

        # Collect image files recursively and label them based on path content
        for root, _, files in os.walk(source_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    all_files_train.append(os.path.join(root, file))
                    if "0_real" in os.path.join(root, file):
                        all_y_train.append(0)
                    else:
                        all_y_train.append(1)

        datasets = os.listdir(self.route + "/test/")
        all_test = []
        all_y_test = []

        for index, dataset in enumerate(datasets):
            test = []
            y_test = []
            direct = self.route + "/test/" + dataset + '/'

            lista1 = (
                glob(direct + '1_fake/*.PNG') + glob(direct + '1_fake/*.png') +
                glob(direct + '1_fake/*.jpg') + glob(direct + '1_fake/*.jpeg')
            )
            test.extend(lista1)
            y_test.extend([1] * len(lista1))

            lista = (
                glob(direct + '0_real/*.png') + glob(direct + '0_real/*.PNG') +
                glob(direct + '0_real/*.jpg') + glob(direct + '0_real/*.jpeg')
            )
            test.extend(lista)
            y_test.extend([0] * len(lista))

            all_test.append(test)
            all_y_test.append(y_test)

        train, val, y_train, y_val = train_test_split(
            all_files_train, all_y_train, train_size=0.8, stratify=all_y_train, random_state=5
        )
        return train, y_train, val, y_val, all_test, all_y_test


def create_and_save_ALL_embeddings_Mamba(model, train_dataloader, path, save, device1):
    """
    Compute embeddings using a Mamba model for all samples in train_dataloader.

    Returns:
    - embeddings: np.array of embeddings
    - labels: np.array of labels
    """
    labels = []
    embeddings = []
    index = 0
    model.eval()
    for image1, label in tqdm(train_dataloader, desc=f"Getting embedding {index + 1}/{len(train_dataloader)}"):
        index += 1
        image1 = image1.to(device1)

        embs = model(image1)['logits']
        embs = embs.detach().cpu()
        label = label.numpy()
        embeddings += embs
        labels.extend(label)

        del embs, label

        gc.collect()
        torch.cuda.empty_cache()

    labels = np.array(labels)
    embeddings = np.array(embeddings)
    if save:
        np.savez(path, embeddings=embeddings, labels=labels)

    return embeddings, labels


def compute_oscr_5classes(all_labels, probs, unseen_label=10):
    """
    Compute OSCR (Open Set Classification Rate) with support for arbitrary label values.

    Parameters
    ----------
    all_labels : array-like, shape (N,)
        Ground truth labels. Unseen/open-set label(s) should match unseen_label.
    probs : array-like, shape (N, K)
        Predicted probabilities for K classes (K includes the open/unseen class column if present).
    unseen_label : int or list/tuple
        Label value(s) representing unseen/unknown class (default 10).

    Returns
    -------
    oscr : float
        Area under CCR vs FPR curve.
    ccr_list : list
        CCR values across thresholds.
    fpr_list : list
        FPR values across thresholds.
    thresholds : list
        Thresholds evaluated.
    """
    # Support list or scalar unseen_label
    if isinstance(unseen_label, (list, tuple)):
        unseen_mask = np.isin(all_labels, unseen_label)
    else:
        unseen_mask = all_labels == unseen_label

    seen_mask = ~unseen_mask

    # Unique seen labels and mapping to contiguous indices
    unique_seen_labels = np.unique(all_labels[seen_mask])
    num_known_classes = len(unique_seen_labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_seen_labels)}

    # Debug prints to help diagnose label/prob shape issues
    print(f"Debug info:")
    print(f"  Unique seen labels: {unique_seen_labels}")
    print(f"  Label to index mapping: {label_to_idx}")
    print(f"  Seen samples: {np.sum(seen_mask)}")
    print(f"  Unseen samples: {np.sum(unseen_mask)}")
    print(f"  Probs shape: {probs.shape}")

    # Seen partition
    seen_labels = all_labels[seen_mask]
    seen_probs = probs[seen_mask]

    # Map seen labels to 0..(num_known_classes-1)
    seen_labels_mapped = np.array([label_to_idx[label] for label in seen_labels])

    # Predictions and max probabilities on seen classes (consider only first num_known_classes columns)
    seen_predictions = np.argmax(seen_probs[:, :num_known_classes], axis=1)
    seen_max_probs = np.max(seen_probs[:, :num_known_classes], axis=1)

    print(f"  Seen max probs range: [{seen_max_probs.min():.3f}, {seen_max_probs.max():.3f}]")

    # Unseen partition
    unseen_probs = probs[unseen_mask]
    unseen_max_probs = np.max(unseen_probs[:, :num_known_classes], axis=1)

    print(f"  Unseen max probs range: [{unseen_max_probs.min():.3f}, {unseen_max_probs.max():.3f}]")
    print(f"  Mean seen prob: {seen_max_probs.mean():.3f}")
    print(f"  Mean unseen prob: {unseen_max_probs.mean():.3f}")

    n_seen = len(seen_labels)
    n_unseen = np.sum(unseen_mask)

    if n_seen == 0 or n_unseen == 0:
        print("ERROR: Need both seen and unseen samples!")
        return 0.0, [], [], []

    # Determine thresholds from combined seen/unseen max probabilities (descending)
    all_probs = np.concatenate([seen_max_probs, unseen_max_probs])
    thresholds = np.sort(np.unique(all_probs))[::-1]
    thresholds = np.concatenate([[1.0], thresholds, [0.0]])

    ccr_list = []
    fpr_list = []

    for tau in thresholds:
        # CCR: correct classification rate among seen examples above threshold
        correct_and_confident = (seen_predictions == seen_labels_mapped) & (seen_max_probs > tau)
        ccr = np.sum(correct_and_confident) / n_seen if n_seen > 0 else 0.0

        # FPR: proportion of unseen examples whose max prob >= tau (i.e., false positives)
        false_positives = np.sum(unseen_max_probs >= tau)
        fpr = false_positives / n_unseen if n_unseen > 0 else 0.0

        ccr_list.append(ccr)
        fpr_list.append(fpr)

    # Compute OSCR as area under CCR vs FPR curve
    if len(fpr_list) > 1 and len(ccr_list) > 1:
        oscr = auc(fpr_list, ccr_list)
    else:
        oscr = 0.0

    print(f"  OSCR computed: {oscr:.4f}")
    print(f"  CCR range: [{min(ccr_list):.3f}, {max(ccr_list):.3f}]")
    print(f"  FPR range: [{min(fpr_list):.3f}, {max(fpr_list):.3f}]")

    return oscr, ccr_list, fpr_list, thresholds