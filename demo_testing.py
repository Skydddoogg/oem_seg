import os
import time
import rasterio
import warnings
import numpy as np
import torch
import cv2
import oem
from pathlib import Path
import segmentation_models_pytorch as smp
import argparse
from tqdm import tqdm

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()

    start = time.time()

    BACKBONE = 'resnet18'
    
    OEM_DATA_DIR = "OpenEarthMap_Mini"
    TEST_LIST = os.path.join(OEM_DATA_DIR, "test.txt")

    N_CLASSES = 9
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = 'cpu'
    PREDS_DIR = "predictions"
    os.makedirs(PREDS_DIR, exist_ok=True)

    fns = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "images" in str(f)]
    test_fns = [str(f) for f in fns if f.name in np.loadtxt(TEST_LIST, dtype=str)]

    print("Total samples   :", len(fns))
    print("Testing samples :", len(test_fns))

    test_data = oem.dataset.OpenEarthMapDataset(
        test_fns,
        n_classes=N_CLASSES,
        augm=None,
        testing=True,
    )

    network = smp.UnetPlusPlus(
        encoder_name=BACKBONE,
        in_channels=3,
        classes=N_CLASSES,   
    )

    network = oem.utils.load_checkpoint(
        network,
        model_name=f"{args.model}.pth",
        model_dir="outputs",
    )

    network.eval().to(DEVICE)
    for idx in tqdm(range(len(test_fns))):
        img, fn = test_data[idx][0], test_data[idx][2]

        with torch.no_grad():
            prd = network(img.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
        prd = oem.utils.make_rgb(np.argmax(prd.numpy(), axis=0))
        fout = os.path.join(PREDS_DIR, fn.split("\\")[-1].split('.')[0] + '.png')

        with rasterio.open(fn, "r") as src:
            profile = src.profile
            prd = cv2.resize(
                prd,
                (profile["width"], profile["height"]),
                interpolation=cv2.INTER_NEAREST,
            )

            cv2.imwrite(fout, cv2.cvtColor(prd, cv2.COLOR_RGB2BGR))
