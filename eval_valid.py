import torch
from pathlib import Path
import os
import numpy as np
import oem
import argparse
import segmentation_models_pytorch as smp
from tqdm import tqdm

def compute_iou(outputs: torch.Tensor, labels: torch.Tensor):

    SMOOTH = 1e-6

    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()

def load_model(model, backbone):

    network = smp.Unet(
        encoder_name=backbone,
        in_channels=3,
        classes=N_CLASSES,   
    )

    network = oem.utils.load_checkpoint(network, model_name=f"{model}.pth", model_dir="outputs")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()

    N_CLASSES = 9
    CLASSES = ["Unknown", "Bareland", "Rangeland", "Developed space", "Road", "Tree", "Water", "Agriculture land", "Building"]
    class_index = [i for i in range(N_CLASSES)]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1

    OEM_DATA_DIR = "OpenEarthMap_Mini"
    VALID_LIST = os.path.join(OEM_DATA_DIR, "val.txt")

    img_paths = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "images" in str(f)]
    val_fns = [str(f) for f in img_paths if f.name in np.loadtxt(VALID_LIST, dtype=str)]

    print("Total samples   :", len(img_paths))
    print("Validation samples :", len(val_fns))


    val_data = oem.dataset.OpenEarthMapDataset(
        val_fns,
        n_classes=N_CLASSES,
        augm=None,
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=False,
    )

    BACKBONE = 'efficientnet-b4'
    network = smp.Unet(
        encoder_name=BACKBONE,
        in_channels=3,
        classes=N_CLASSES,   
    )

    network = oem.utils.load_checkpoint(network, model_name=f"{args.model}.pth", model_dir="outputs")

    network.to(DEVICE).eval()

    class_iou = [[] for c in class_index]

    iterator = tqdm(val_data_loader, desc="Validation")
    for x, y, *_ in iterator:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        n = x.shape[0]

        with torch.no_grad():
            pd = network.forward(x).squeeze(0).cpu()
            
            for c in class_index:
            
                iou = compute_iou(
                    torch.tensor((np.argmax(pd.detach().numpy(), axis=0) == c) * 1)[None, :, :], 
                    ((np.argmax(y.squeeze(0).cpu(), axis=0) == c) * 1)[None, :, :]
                )
                
                class_iou[c] += [iou]

    class_iou = torch.tensor(class_iou)
    print(f'mIoU = {class_iou[1:].mean(axis=1).mean() * 100:.2f}')
    for i,c in enumerate(CLASSES):
        print(f'  - {c} = {class_iou[i].mean() * 100:.2f}')