import torch
from torch.utils.data import DataLoader
import gc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoModelForImageClassification
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import numpy as np
import yaml
from utils.data_acquisition import test_dataset, data_set_binary_with_nature, create_and_save_ALL_embeddings_Mamba

# Entry point for running this script directly
if __name__ == "__main__":
    
    # Load configuration values from YAML
    with open("tests/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Read values from config
    BATCH_SIZE = config["BATCH_SIZE"]
    RESOLUTION = config["RESOLUTION"]
    device = config["DEVICE"]
    ROUTE_ENCODER = config["PATHS"]["ROUTE_ENCODER"]
    k_instances = config["KNN_INSTANCES"]
    split = config["SPLIT"]

    # Prepare dataset loader that knows how to return train/test groups
    loader_data = data_set_binary_with_nature('Datasets/GenImage_sampled/')
    all_train, all_test, all_y_train, all_y_test = loader_data.get_data(split)
    
    # Number of generators (classes) in the original grouping
    n_generators = len(all_train) + 1

    # Flatten nested lists of samples/labels into single numpy arrays for selection
    all_train = np.array([x for sublist in all_train for x in sublist])
    all_y_train = np.array([x for sublist in all_y_train for x in sublist])

    # k_real is used to compensate the real class (value found via grid search)
    k_real = k_instances * 2

    data_selected = []
    y_selected = []

    # Sample k_instances examples per generator/class (with special handling for class index 8)
    for i in range(n_generators):
        if i != 8:
            indices = np.random.choice(np.argwhere(all_y_train == i).flatten(), size=k_instances, replace=False)
        else:
            indices = np.random.choice(np.argwhere(all_y_train == i).flatten(), size=k_real, replace=False)
        data_selected.extend(all_train[indices])
        y_selected.extend(all_y_train[indices])
    
    # Convert labels to binary: class 8 -> 0 (real), others -> 1 (fake)
    for i in range(len(y_selected)):
        if y_selected[i] == 8:
            y_selected[i] = 0
        else:
            y_selected[i] = 1

    # Load pretrained image classification model and then load encoder checkpoint weights
    model = AutoModelForImageClassification.from_pretrained(
        "nvidia/MambaVision-L3-256-21K",
        trust_remote_code=True,
        dtype="auto"
    ).to("cpu")

    # Load checkpoint (encoder weights)
    checkpoint = torch.load(ROUTE_ENCODER, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Create a dataset and dataloader for the selected training samples (used for KNN training)
    train_dataset = test_dataset(data_selected, y_selected, device, RESOLUTION)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        prefetch_factor=2
    )

    # Compute and (optionally) save embeddings for the training set using the model
    train_embeddings, label_embeddings = create_and_save_ALL_embeddings_Mamba(
        model,
        train_dataloader=train_dataloader,
        path=None,
        save=False,
        device1=device
    )

    # Fit a KNN classifier on computed embeddings
    knn = KNeighborsClassifier(n_neighbors=11, n_jobs=12)
    knn.fit(train_embeddings, label_embeddings)

    # Iterate over each group in the test split (each generator vs real)
    for i in range(len(all_test)):

        all_roc = []
        all_acc = []

        temp_test = all_test[i]
        temp_y_test = all_y_test[i]

        print("Creating dataloaders")
        test_data = test_dataset(temp_test, temp_y_test, device, RESOLUTION)
        test_dataloader = DataLoader(
            test_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            prefetch_factor=2
        )

        # Accumulators for true labels and predicted probabilities
        all_labels = []
        all_preds = []
            
        suma = 0
        total = 0
        index = 0

        # Iterate batches, compute embeddings, predict with KNN
        for image1, label in tqdm(test_dataloader, desc=f"Classifying {index + 1}/{len(test_dataloader)}"):
            with torch.no_grad():
                index += 1

                # Move the entire batch to the device
                image1 = image1.to(device)

                # Model returns logits; here logits are used as embeddings
                embeddings = model(image1)['logits']
                embeddings = embeddings.cpu().numpy()

                # KNN expects numpy arrays; get predicted probabilities for each class
                predictions = knn.predict_proba(embeddings)

                # Convert label tensor to numpy and store results
                label = label.cpu().numpy()
                all_labels.extend(label)
                all_preds.extend(predictions)
                
                total += len(label)

                # Free memory after each batch
                del image1, label, predictions, embeddings
                gc.collect()
                torch.cuda.empty_cache()

        # Convert accumulators to numpy arrays for metric computation
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # Predicted class is index with highest probability
        pr = np.array([np.argmax(np.array(pred)) for pred in all_preds])

        # Convert ground-truth: label 8 -> 0 (real), else 1 (fake)
        labels = np.array([0 if label == 8 else 1 for label in all_labels])

        # Compute ROC AUC for both conventions (sometimes class ordering flips)
        roc1 = roc_auc_score(labels, 1 - pr)
        roc = roc_auc_score(labels, pr)
        if roc1 > roc:
            roc = roc1

        print(f"ROC AUC for class {i} vs Real: {roc:.4f}")

        accuracy = accuracy_score(labels, pr)
        print(f"Accuracy for class {i} vs Real: {accuracy:.4f}")


