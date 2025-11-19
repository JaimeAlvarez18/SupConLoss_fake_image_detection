import yaml
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import math
import gc
import torch.amp as amp
from transformers import AutoModelForImageClassification
import numpy as np
from einops import rearrange
from utils.data_acquisition import data_set_with_nature, BIG_DATALOADER, read_files, data_set_binary_synth
from utils.loss import SupConLoss

# Script entry point: training a supervised contrastive model using a pretrained "Mamba" vision encoder.
# This file only adds comments and minor formatting; no logic was changed.

if __name__ == "__main__":

    # Load training configuration
    with open("training/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Config values used throughout training
    BATCH_SIZE = config["BATCH_SIZE"]
    RESOLUTION = config["RESOLUTION"]
    device = config["DEVICE"]
    PATH_TO_SAVE_Epoch = config["PATHS"]["SAVE_EPOCH"]

    # Parameters controlling minibatch splitting to limit memory use
    MAX_MINIBATCH_PROCESS = config["MAX_MINIBATCH_PROCESS"]
    SMALL_MINIBATCH = config["SMALL_MINIBATCH"]

    EPOCHS = config["EPOCHS"]
    dataset = config["DATASET"]
    split = config["SPLIT"]

    print("Getting data ...")

    # Select dataset loader based on config
    if dataset == "GenImage":
        loader_data = data_set_with_nature('Datasets/GenImage_sampled/')
        train, val, test, y_train, y_val, y_testl, _, _ = loader_data.get_data(split)
    elif dataset == "ForenSynths":
        loader_data = data_set_binary_synth('Datasets/ForenSynths_sampled/')
        train, y_train, val, y_val, _, _ = loader_data.get_data()

    print("Creating Dataloaders ...")
    # BIG_DATALOADER returns raw file path + label pairs; DataLoader will iterate them.
    train_dataset = BIG_DATALOADER(train, y_train, device, RESOLUTION)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, prefetch_factor=1)

    val_dataset = BIG_DATALOADER(val, y_val, device, RESOLUTION)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, prefetch_factor=1)

    # Free references to lists (datasets already wrapped in dataloaders)
    del train, y_train, val, y_val, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

    best = math.inf
    print(f"Creating model...")
    batch_losses = []

    # Load pretrained Mamba model and prepare optimizer / loss / scaler
    model = AutoModelForImageClassification.from_pretrained(
        "nvidia/MambaVision-L3-256-21K", trust_remote_code=True, dtype="auto"
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = SupConLoss()
    scaler = amp.GradScaler()

    print("Training model...")
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    # Main training loop over epochs
    for epoch in range(EPOCHS):
        model.train()  # Enable training behavior (if any)

        # Per-epoch accumulators
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate over training batches of file paths + labels
        for index, (routes, labels) in tqdm(enumerate(train_dataloader), desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):

            # Save a snapshot of the current model weights in case validation does not improve
            previous = model.state_dict()

            routes = np.array(list(routes))
            optimizer.zero_grad()

            # Split the current batch into chunks to limit memory consumption when reading files
            chunk_size = math.ceil(routes.shape[0] / MAX_MINIBATCH_PROCESS)
            chunks_routes = np.array_split(routes, chunk_size)
            chunks_labels = np.array_split(labels, chunk_size)

            # Extract embeddings for each chunk without gradient computation
            all_embeddings = []
            for batch_routes in chunks_routes:
                data = read_files(batch_routes)        # read_files returns a torch tensor of images
                data = data.to(device)
                with torch.no_grad():
                    with amp.autocast(device_type='cuda'):
                        pred1 = model(data)['logits']
                        pred1 = pred1.unsqueeze(1)
                    pred1 = pred1.to('cpu')
                    all_embeddings.extend(pred1)
                # release per-chunk tensors
                del data, pred1
                gc.collect()
                torch.cuda.empty_cache()

            # Prepare minibatches for contrastive loss computation
            loss_tracker = []
            all_embeddings = torch.from_numpy(np.array(all_embeddings)).unsqueeze(0).to('cpu')
            # Rearrange embeddings to shape expected by loss
            all_embeddings = rearrange(all_embeddings, 'b n k c -> (b n) k c')

            # Now split routes/labels into small minibatches for gradient updates
            num_chunks = math.ceil(routes.shape[0] / SMALL_MINIBATCH)
            chunks_routes_small = np.array_split(routes, num_chunks)
            chunks_labels_small = np.array_split(labels, num_chunks)

            current_index = 0  # index into the precomputed all_embeddings tensor

            # Iterate over the small minibatches, compute embeddings for the small batch,
            # substitute them into 'rep', compute loss and backpropagate.
            for index1, (batch_routes, batch_labels) in enumerate(zip(chunks_routes_small, chunks_labels_small)):
                rep = all_embeddings.clone()

                data = read_files(batch_routes)
                data = data.to(device)
                with amp.autocast(device_type='cuda'):
                    preds = model(data)['logits'].unsqueeze(1).to('cpu')
                    # Replace corresponding slice in rep with the freshly computed preds
                    rep[current_index: current_index + preds.shape[0]] = preds
                    current_index += preds.shape[0]

                    # free input tensor
                    del data
                    gc.collect()
                    torch.cuda.empty_cache()

                    labels = labels.to(device)
                    rep = rep.to(device)

                    # Compute supervised contrastive loss on 'rep' and labels
                    loss = criterion(rep, labels)

                # Cleanup for this small minibatch
                del rep, preds
                gc.collect()
                torch.cuda.empty_cache()

                loss_tracker.append(loss.item())

                # Backpropagate scaled loss (using GradScaler for mixed precision)
                scaler.scale(loss).backward()
                del loss
                gc.collect()
                torch.cuda.empty_cache()

            # Step the optimizer and update scaler after processing all small minibatches
            scaler.step(optimizer)
            scaler.update()
            loss = sum(loss_tracker) / len(loss_tracker)

            # Free batch-level references
            del routes, labels, loss_tracker
            gc.collect()
            torch.cuda.empty_cache()

            batch_losses.append(loss)

        # Compute epoch training loss (mean of batch losses)
        train_loss_value = np.mean(np.array(batch_losses))

        # Validation loop: same pattern as training but without optimizer updates
        for index, (routes, labels) in tqdm(enumerate(val_dataloader), desc=f"Validating Epoch {epoch + 1}/{EPOCHS}"):

            routes = np.array(list(routes))
            chunk_size = math.ceil(routes.shape[0] / MAX_MINIBATCH_PROCESS)
            chunks_routes = np.array_split(routes, chunk_size)
            chunks_labels = np.array_split(labels, chunk_size)

            all_embeddings = []
            for batch_routes in chunks_routes:
                data = read_files(batch_routes)
                data = data.to(device)
                with torch.no_grad():
                    with amp.autocast(device_type='cuda'):
                        pred1 = model(data)['logits']
                        pred1 = pred1.unsqueeze(1)
                    pred1 = pred1.to('cpu')
                    all_embeddings.extend(pred1)
                del data, pred1
                gc.collect()
                torch.cuda.empty_cache()

            loss_tracker = []
            all_embeddings = torch.from_numpy(np.array(all_embeddings)).unsqueeze(0).to('cpu')
            all_embeddings = rearrange(all_embeddings, 'b n k c -> (b n) k c')

            num_chunks = math.ceil(routes.shape[0] / SMALL_MINIBATCH)
            chunks_routes_small = np.array_split(routes, num_chunks)
            chunks_labels_small = np.array_split(labels, num_chunks)

            current_index = 0

            for index1, (batch_routes, batch_labels) in enumerate(zip(chunks_routes_small, chunks_labels_small)):
                rep = all_embeddings.clone()

                data = read_files(batch_routes)
                data = data.to(device)
                with torch.no_grad():
                    with amp.autocast(device_type='cuda'):
                        preds = model(data)['logits'].unsqueeze(1).to('cpu')
                        rep[current_index: current_index + preds.shape[0]] = preds
                        current_index += preds.shape[0]

                        del data
                        gc.collect()
                        torch.cuda.empty_cache()

                        labels = labels.to(device)
                        rep = rep.to(device)

                        loss = criterion(rep, labels)
                del rep, preds

                gc.collect()
                torch.cuda.empty_cache()
                loss_tracker.append(loss.item())

                del loss
                gc.collect()
                torch.cuda.empty_cache()

            loss = sum(loss_tracker) / len(loss_tracker)

            # Free validation batch-level references
            del routes, labels, loss_tracker
            gc.collect()
            torch.cuda.empty_cache()

            batch_losses.append(loss)

            # Save checkpoint: if validation improved use current model, else use previous snapshot
            if loss < best:
                best = loss
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best,
                    "batch_losses": batch_losses
                }
            else:
                checkpoint = {
                    "model_state_dict": previous,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best,
                    "batch_losses": batch_losses
                }

            torch.save(checkpoint, PATH_TO_SAVE_Epoch)
            del loss
            gc.collect()
            torch.cuda.empty_cache()

        val_loss_value = np.mean(np.array(batch_losses))

        # Epoch summary
        print('-' * 60)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss_value:.4f}")
        print(f"Val Loss: {val_loss_value:.4f}")
        print('-' * 60)
        print()