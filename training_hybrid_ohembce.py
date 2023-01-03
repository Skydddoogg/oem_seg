import os
import time
import warnings
import numpy as np
import torch
import oem
import torchvision
from pathlib import Path
import argparse
import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()

    start = time.time()

    BACKBONE = 'resnet18'
    PRETRAINED = 'imagenet'

    OEM_DATA_DIR = "OpenEarthMap_Mini"
    TRAIN_LIST = os.path.join(OEM_DATA_DIR, "train.txt")
    VAL_LIST = os.path.join(OEM_DATA_DIR, "val.txt")

    IMG_SIZE = 512
    N_CLASSES = 9
    LR = 0.001
    BATCH_SIZE = 4
    NUM_EPOCHS = 150
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_paths = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "images" in str(f)]
    train_fns = [str(f) for f in img_paths if f.name in np.loadtxt(TRAIN_LIST, dtype=str)]
    val_fns = [str(f) for f in img_paths if f.name in np.loadtxt(VAL_LIST, dtype=str)]

    print("Total samples      :", len(img_paths))
    print("Training samples   :", len(train_fns))
    print("Validation samples :", len(val_fns))

    train_augm = torchvision.transforms.Compose(
        [
            # oem.transforms.Resize(IMG_SIZE),
            oem.transforms.Rotate(),
            oem.transforms.Crop(IMG_SIZE),
        ],
    )

    val_augm = torchvision.transforms.Compose(
        [
            oem.transforms.Resize(IMG_SIZE),
        ],
    )

    train_data = oem.dataset.OpenEarthMapDataset(
        train_fns,
        n_classes=N_CLASSES,
        augm=train_augm,
    )

    val_data = oem.dataset.OpenEarthMapDataset(
        val_fns,
        n_classes=N_CLASSES,
        augm=val_augm,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=True,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=False,
    )

    network = smp.Unet(
        encoder_name=BACKBONE,
        encoder_weights=PRETRAINED,
        in_channels=3,
        classes=N_CLASSES,   
    )

    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    criterion = oem.losses.HybridOHEMBCELoss()

    train_loss = []
    val_loss = []
    max_score = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch + 1}")

        train_logs = oem.runners.train_epoch(
            model=network,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=DEVICE,
        )

        valid_logs = oem.runners.valid_epoch(
            model=network,
            criterion=criterion,
            dataloader=val_data_loader,
            device=DEVICE,
        )

        epoch_score = valid_logs["Score"]
        if max_score < epoch_score:
            max_score = epoch_score
            oem.utils.save_model(
                model=network,
                epoch=epoch,
                best_score=max_score,
                model_name=f"{args.model}.pth",
                output_dir=OUTPUT_DIR,
            )

        train_loss += [train_logs['Loss']]
        val_loss += [valid_logs['Loss']]

    oem.utils.save_loss(train_loss, save_path=os.path.join(OUTPUT_DIR, f'{args.model}_train_loss.txt'))
    oem.utils.save_loss(val_loss, save_path=os.path.join(OUTPUT_DIR, f'{args.model}_val_loss.txt'))
    print(f'Best IoU on validation set = {max_score:.2f}')
    print("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))
