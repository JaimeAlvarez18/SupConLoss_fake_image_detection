import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
import gc
from utils.data_acquisition import (
    create_and_save_ALL_embeddings_Mamba,
    test_dataset,
    data_set_with_nature,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from transformers import AutoModelForImageClassification
from tqdm import tqdm
import yaml
import numpy as np

def print_confusion_matrix(cm, labels=list(range(9))):
    """
    Pretty-print a confusion matrix with row/column labels.

    Args:
        cm (np.ndarray): square confusion matrix.
        labels (list): list of labels to display for rows/columns.
    """
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    # Ensure labels are strings for formatting
    labels = [str(l) for l in labels]

    # Header and column width based on label lengths
    header = [""] + labels
    col_width = max(max(len(x) for x in labels), 5)

    # Print header
    print("\nConfusion Matrix:")
    print(" " * (col_width + 2) + "Predicted")
    print(" " * (col_width + 2) + " ".join(f"{h:>{col_width}}" for h in labels))

    # Print each row with its label
    for i, row in enumerate(cm):
        row_label = labels[i]
        row_values = " ".join(f"{val:>{col_width}}" for val in row)
        print(f"{row_label:>{col_width}} | {row_values}")
    print()


if __name__ == '__main__':
    # Load configuration values from tests/config.yaml
    with open("tests/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Read settings from config
    BATCH_SIZE = config["BATCH_SIZE"]
    RESOLUTION = config["RESOLUTION"]
    device = config["DEVICE"]
    split = config["SPLIT"]
    ROUTE_ENCODER = config["PATHS"]["ROUTE_ENCODER"]
    k_instances = config["KNN_INSTANCES"]

    # Prepare dataset loader that includes nature/real samples
    loader_data = data_set_with_nature('Datasets/GenImage_sampled/')
    train, val, test, y_train, y_val, y_test, train_indices, test_indices = loader_data.get_data(split, True)

    # Number of generators expected in this setup
    n_generators = 9

    # Load pretrained vision model (Mamba) and then load encoder checkpoint weights
    model = AutoModelForImageClassification.from_pretrained(
        'nvidia/MambaVision-L3-256-21K',
        trust_remote_code=True,
        dtype='auto'
    ).to('cpu')

    checkpoint = torch.load(ROUTE_ENCODER, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # evaluation mode (disables dropout, etc.)

    # Select a subset of training data for KNN training.
    # For generators present in test_indices we sample fewer examples (their sum are the same as training generators),
    # otherwise sample k_instances examples per generator.
    data_selected = []
    y_selected = []
    for i in range(n_generators):
        if i not in test_indices:
            # Regular generators: sample k_instances examples
            indices = np.random.choice(np.argwhere(y_train == i).flatten(), size=k_instances, replace=False)
        else:
            # Generators in test_indices: sample fewer examples (downscaled)
            indices = np.random.choice(
                np.argwhere(y_train == i).flatten(),
                size=int(k_instances / len(test_indices)),
                replace=False
            )

        data_selected.extend(np.array(train)[indices])
        y_selected.extend(np.array(y_train)[indices])

    # Build dataloaders for training (KNN fitting) and testing
    train_dataset = test_dataset(data_selected, y_selected, device, RESOLUTION)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        prefetch_factor=1
    )

    test_data = test_dataset(test, y_test, device, RESOLUTION)
    test_dataloader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        prefetch_factor=1
    )

    # Compute embeddings for training data using the model (function returns embeddings and labels)
    train_embeddings, label_embeddings = create_and_save_ALL_embeddings_Mamba(
        model,
        train_dataloader=train_dataloader,
        path=None,
        save=False,
        device1=device
    )

    # Fit a KNN classifier on the embeddings
    knn = KNeighborsClassifier(n_neighbors=11, n_jobs=-1)
    knn.fit(train_embeddings, label_embeddings)

    # Cleanup GPU memory before inference
    gc.collect()
    torch.cuda.empty_cache()

    # Inference loop: compute model logits for test batches and predict with KNN
    sum = 0
    total = 0
    index = 0
    all_labels = []
    all_preds = []

    for image1, label in tqdm(test_dataloader, desc=f'Classifying {index + 1}/{len(test_dataloader)}'):
        index += 1
        with torch.no_grad():
            # Move batch to device and compute logits (used as embeddings)
            image1 = image1.to(device)
            outputs = model(image1)['logits'].cpu().numpy()

            # KNN predicts class indices from embeddings
            predictions = knn.predict(outputs)

            # Accumulate true labels and predictions
            all_labels.extend(label)
            all_preds.extend(predictions)

            # Track accuracy counts
            sum += np.sum(np.array(label) == np.array(predictions))
            total += len(label)

            # Free memory for next batch
            del image1, label, predictions, outputs
            gc.collect()
            torch.cuda.empty_cache()

    # Compute overall accuracy and confusion matrix
    accuracy = (sum / total) * 100
    cm = confusion_matrix(all_labels, all_preds)

    # Compute per-class and weighted precision/recall/f1 scores
    precision1 = precision_score(all_labels, all_preds, labels=test_indices, average=None)
    recall1 = recall_score(all_labels, all_preds, labels=test_indices, average=None)
    f11 = f1_score(all_labels, all_preds, labels=test_indices, average=None)

    precision2 = precision_score(all_labels, all_preds, labels=test_indices, average="weighted")
    recall2 = recall_score(all_labels, all_preds, labels=test_indices, average="weighted")
    f12 = f1_score(all_labels, all_preds, labels=test_indices, average="weighted")

    precision3 = precision_score(all_labels, all_preds, labels=train_indices, average="weighted")
    recall3 = recall_score(all_labels, all_preds, labels=train_indices, average="weighted")
    f13 = f1_score(all_labels, all_preds, labels=train_indices, average="weighted")

    # Print summary metrics and confusion matrix
    print("-" * 100)
    print("-" * 100)
    print("Precision zero-shot:", ", ".join(f"{x:.4f}" for x in precision1))
    print("Recall zero-shot:", ", ".join(f"{x:.4f}" for x in recall1))
    print("F1-score zero-shot:", ", ".join(f"{x:.4f}" for x in f11))
    print("-" * 100)
    print(f"Total precis zero-shot: {precision2:.4f}")
    print(f"Total recall zero-shot: {recall2:.4f}")
    print(f"Total F1-sco zero-shot: {f12:.4f}")
    print("-" * 100)
    print(f"Total precis NO zero-shot: {precision3:.4f}")
    print(f"Total recall NO zero-shot: {recall3:.4f}")
    print(f"Total F1-sco NO zero-shot: {f13:.4f}")
    print("-" * 100)
    print("-" * 100)

    print()
    print('-' * 60)
    print('-' * 60)
    print("Test")
    print(f"Accuracy: {accuracy}.")
    print(f"F1: {f12}.")
    print(f"Validation Confusion Matrix:")
    print_confusion_matrix(cm)
    print('-' * 60)
    print('-' * 60)
    print()