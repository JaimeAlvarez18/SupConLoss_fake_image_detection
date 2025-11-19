import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
import gc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  roc_auc_score, accuracy_score
from transformers import AutoModelForImageClassification
from utils.data_acquisition import compute_oscr_5classes,test_dataset, data_set_with_nature,create_and_save_ALL_embeddings_Mamba
from tqdm import tqdm
import numpy as np
import yaml

if __name__ == '__main__':
    # Load test configuration
    with open("tests/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Read values from config
    BATCH_SIZE = config["BATCH_SIZE"]
    RESOLUTION = config["RESOLUTION"]
    device = config["DEVICE"]
    split = config["SPLIT"]
    ROUTE_ENCODER = config["PATHS"]["ROUTE_ENCODER"]
    k_instances = config["KNN_INSTANCES"]

    # Prepare dataset loader (includes "nature"/real class)
    loader_data = data_set_with_nature('Datasets/GenImage_sampled/')
    train, val, test, y_train, y_val, y_test, train_indices, test_indices = loader_data.get_data(split,True)

    # Number of source generators (classes)
    n_generators = 9

    # Load pretrained Mamba vision model and then load checkpoint weights (encoder)
    model = AutoModelForImageClassification.from_pretrained('nvidia/MambaVision-L3-256-21K', trust_remote_code=True, dtype='auto').to('cpu')
    checkpoint = torch.load(ROUTE_ENCODER, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # set model to evaluation mode

    # Select subset of training samples for KNN fitting.
    # For classes in test_indices we downscale the number of selected samples to simulate open-set.
    data_selected = []
    y_selected = []
    
    for i in range(n_generators):
        if i not in test_indices:
            # Regular generators: sample k_instances samples
            indices = np.random.choice(np.argwhere(y_train == i).flatten(), size=k_instances, replace=False)
        else:
            # Test generators: sample fewer examples (downscaled by number of test indices)
            indices = np.random.choice(np.argwhere(y_train == i).flatten(), size=int(k_instances / len(test_indices)), replace=False)
        
        data_selected.extend(np.array(train)[indices])
        y_selected.extend(np.array(y_train)[indices])

    # Relabel selected and full train/test labels: test generators become label 10 (unknown/open)
    y_selected = [10 if i in test_indices else i for i in y_selected]
    y_train = [10 if i in test_indices else i for i in y_train]
    y_test = [10 if i in test_indices else i for i in y_test]

    # Build training dataloader (for embedding extraction) and testing dataloader
    train_dataset = test_dataset(data_selected, y_selected, device, RESOLUTION)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, prefetch_factor=1)

    test_data = test_dataset(test, y_test, device, RESOLUTION)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, prefetch_factor=1)

    # Compute embeddings for selected training samples using the model (returns embeddings and labels)
    train_embeddings, label_embeddings = create_and_save_ALL_embeddings_Mamba(model, train_dataloader=train_dataloader, path=None, save=False, device1=device)

    # Fit KNN classifier on training embeddings
    knn = KNeighborsClassifier(n_neighbors=11, n_jobs=-1)
    knn.fit(train_embeddings, label_embeddings)

    # Free any unused GPU memory before inference
    gc.collect()
    torch.cuda.empty_cache()

    # Inference: compute model embeddings for test batches and obtain KNN probabilities
    sum = 0
    total = 0
    index = 0
    all_labels = []
    all_probs = []
    for image1, label in tqdm(test_dataloader, desc=f'Classifying {index + 1}/{len(test_dataloader)}'):
        index += 1
        with torch.no_grad():
            image1 = image1.to(device)
            outputs = model(image1)['logits'].cpu().numpy()  # logits used as embeddings
            label = label.cpu().numpy()
            probs = knn.predict_proba(outputs)  # KNN probabilities for each class (including open class)
            all_labels.extend(label)
            all_probs.extend(probs)

            # Cleanup per-batch tensors to reduce memory usage
            del image1, label, probs, outputs
            gc.collect()
            torch.cuda.empty_cache()

    # Convert accumulators to numpy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Determine valid (closed-set) class indices (exclude open label 10)
    valid_indices = np.unique(np.array(all_labels))
    valid_indices = np.delete(valid_indices, np.where(valid_indices == 10))

    # Compute open-set AUC using all classes (multi-class OVR)
    open_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    print(f'Open AUC: {open_auc}')

    # Compute OSCR metric (open-set classification rate) for 5 classes helper
    oscr, _, _, _ = compute_oscr_5classes(all_labels, all_probs)
    print(f'Open OSCR: {oscr}')

    # For closed-set accuracy: filter out open examples and remove the open-class probabilities column
    mask = np.isin(all_labels, valid_indices)
    filtered_labels = all_labels[mask]
    filtered_probs = all_probs[mask][:, :(-1)]  # drop the last column (open class)
    filtered_probs = filtered_probs / filtered_probs.sum(axis=1, keepdims=True)  # renormalize probabilities

    # Remap original label ids to contiguous 0..K-1 for accuracy computation
    label_mapping = {old: new for new, old in enumerate(valid_indices)}
    remapped_labels = np.array([label_mapping[label] for label in filtered_labels])

    # Predicted closed-set labels and closed accuracy
    predicted_labels = np.argmax(filtered_probs, axis=1)
    accuracy = accuracy_score(remapped_labels, predicted_labels)
    print(f'Closed accuracy: {accuracy}')