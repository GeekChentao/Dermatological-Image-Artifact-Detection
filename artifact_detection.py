#!/usr/bin/env python3.12

# copy GroundingDINO and data and weights

# freeze current pip
# pip freeze | grep "" >> requirements.txt

# install libs in requirements
# pip install -r requirements.txt

# This code runs on Ubuntu with GPU and generate the DINO scores for all images in the dataset and save results in <list.txt>

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import sys
from tqdm import tqdm
import warnings


class PathDataset(Dataset):
    def __init__(self, image_path_list, parent_dir, transform=None) -> None:
        self.path_list = image_path_list
        self.dir = parent_dir
        self.transform = transform

    def __getitem__(self, idx) -> str:
        extension = ".jpg"
        image_path = os.path.join(self.dir, self.path_list[idx]) + extension
        return image_path

    def __len__(self):
        return len(self.path_list)


image_dir = os.path.abspath(os.path.join(os.getcwd(), ".", "data", "Fitzpatric_subset"))
data_csv = "./data/data.csv"
data = pd.read_csv(data_csv)
path_list = data["image_path"].tolist()

# accessories_dir = os.path.abspath(os.path.join(os.getcwd, "data", "Accessories"))
# accessories_list = list()
for i in range(5):
    # accessories_list.append(os.path.join(accessories_dir, f"{i}.jpg"))
    path_list.append(f"{i}")


# path_list = path_list + accessories_list

path_dataset = PathDataset(path_list, image_dir)
path_loader = DataLoader(path_dataset, batch_size=32, shuffle=False, num_workers=4)
print(f"Total {len(path_dataset)} images in the dataset")
print("Abs path: ", image_dir)

# import_path = "../GroundingDINO"
# if import_path not in sys.path:
#     sys.path.insert(0, import_path)

from groundingdino.util.inference import load_model, load_image, predict, annotate

loaded = False

if not loaded:
    model = load_model(
        "./GroundingDINO_SwinT_OGC.py",
        "./weights/groundingdino_swint_ogc.pth",
    )
    print("Model loaded")
    loaded = True


def dino_detect(
    image_path, caption="skin", box_threshold=0.3, text_threshold=0.25, device="cuda"
):

    image_nparray, image_transformed = load_image(image_path)

    print(f"box threshold = {box_threshold}, text threshold = {text_threshold}")

    boxes, logits, phrases = predict(
        device=device,
        model=model,
        image=image_transformed,
        caption=caption,
        # fine tune the thresholds
        box_threshold=box_threshold,  # Confidence score for the object detector: Higher value = fewer, more confident boxes, Lower value = more boxes, but potentially more false positives
        text_threshold=text_threshold,  # Text–box alignment confidence: Higher value = stronger text-to-region match, Lower value = allows weaker associations to pass
    )

    if phrases or logits.numel():
        result = "Found following objects:\n"
        for p in phrases:
            result += f"{p}\n"
        output = f"{image_path}\n{result} logits: {logits}"
        print(output, file=sys.stdout)
        return output
    else:
        return None


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

artifacts = [
    "pen",
    "sticker",
    "marker",
    "cloth",
    "jewelry",
    "watch",
    "shadow",
    "light",
    "hair strands",
    "nail",
    "finger",
    "ear",
    "eye",
    "nostril",
    "lip",
    "glare",
    "blurriness",
]

accessories = ["jewelry", "watch"]

caption = ",".join(accessories)

print(f"Detection Caption:\n{caption}")
with open("list.txt", "w") as f:
    for batch in tqdm(path_loader, desc="DINO Processing\n"):
        for path in batch:
            output = dino_detect(
                path, caption=caption, text_threshold=0.6, box_threshold=0.6
            )
            if output:  # Only write to file if output is not None
                print(output, file=f)
