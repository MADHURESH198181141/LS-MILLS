"""
FGNET Dataset Loader
Structure:
    FGNET/images/   → face images named like 001A02.jpg
                       001 = person ID
                       A   = always A
                       02  = age at capture
    FGNET/points/   → facial landmark files (not needed for matching)
    FGNET/data_file → metadata (not needed for matching)

We use only the IMAGES folder for training.
Pairs are formed by same person ID with different ages (age gap ≥ 5 years).
"""

import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
from collections import defaultdict
import logging

log = logging.getLogger(__name__)


def parse_fgnet_filename(filename):
    """
    Parse FGNET filename to extract person_id and age.
    Format: 001A02.JPG → person_id=1, age=2
            069A34.JPG → person_id=69, age=34
    """
    name = Path(filename).stem  # e.g. '001A02'
    match = re.match(r"(\d+)[Aa](\d+)", name)
    if match:
        person_id = int(match.group(1))
        age = int(match.group(2))
        return person_id, age
    return None, None


class FGNETDataset(Dataset):
    """
    FGNET Face Dataset for age-variation face matching.
    Returns (anchor_image, positive_image, label) triplets
    where anchor and positive are the same person at different ages.
    """

    def __init__(
        self,
        images_dir: str,
        img_size: int = 112,
        augment: bool = True,
        min_age_gap: int = 5,
        mode: str = "train"   # train or val
    ):
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.augment = augment
        self.min_age_gap = min_age_gap
        self.mode = mode

        # Parse all images
        self.person_images = defaultdict(list)  # {person_id: [(path, age), ...]}
        self.all_samples = []                   # [(path, person_id, age)]
        self.label_to_idx = {}
        self._parse_dataset()

        # Build transform
        self.transform = self._build_transform()

        log.info(
            f"FGNET Dataset ({mode}): "
            f"{len(self.all_samples)} images | "
            f"{len(self.person_images)} identities"
        )

    def _parse_dataset(self):
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        label_idx = 0

        for f in sorted(self.images_dir.iterdir()):
            if f.suffix not in exts:
                continue
            person_id, age = parse_fgnet_filename(f.name)
            if person_id is None:
                continue

            self.person_images[person_id].append((str(f), age))
            self.all_samples.append((str(f), person_id, age))

            if person_id not in self.label_to_idx:
                self.label_to_idx[person_id] = label_idx
                label_idx += 1

        # Sort each person's images by age
        for pid in self.person_images:
            self.person_images[pid].sort(key=lambda x: x[1])

        log.info(f"Total persons: {len(self.person_images)}")
        log.info(f"Total images:  {len(self.all_samples)}")

    def _build_transform(self):
        base = [
            T.ToPILImage(),
            T.Resize((self.img_size, self.img_size)),
        ]
        if self.augment and self.mode == "train":
            base += [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                T.RandomGrayscale(p=0.05),
                T.RandomRotation(degrees=10),
            ]
        base += [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        return T.Compose(base)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        img_path, person_id, age = self.all_samples[idx]
        label = self.label_to_idx[person_id]

        image = cv2.imread(img_path)
        if image is None:
            # Return zero tensor if image unreadable
            return torch.zeros(3, self.img_size, self.img_size), torch.tensor(label)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image)
        return tensor, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self):
        return len(self.label_to_idx)

    def get_pair_samples(self):
        """
        Build verification pairs for evaluation:
        Positive pairs: same person, age gap >= min_age_gap
        Negative pairs: different persons
        """
        positive_pairs = []
        negative_pairs = []

        persons = list(self.person_images.keys())

        # Positive pairs
        for pid, imgs in self.person_images.items():
            if len(imgs) < 2:
                continue
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    path1, age1 = imgs[i]
                    path2, age2 = imgs[j]
                    if abs(age1 - age2) >= self.min_age_gap:
                        positive_pairs.append((path1, path2, 1))  # 1=same person

        # Negative pairs — equal count as positive (fixes 81 vs 3793 imbalance)
        import random
        random.seed(42)
        random.shuffle(persons)
        n = len(persons)
        target = len(positive_pairs)

        for i in range(target):
            pid1 = persons[i % n]
            pid2 = persons[(i + 1) % n]
            if pid1 == pid2:
                pid2 = persons[(i + 2) % n]
            imgs1 = self.person_images[pid1]
            imgs2 = self.person_images[pid2]
            if imgs1 and imgs2:
                negative_pairs.append((imgs1[0][0], imgs2[0][0], 0))

        log.info(f"Verification pairs — Positive: {len(positive_pairs)} | Negative: {len(negative_pairs)}")
        return positive_pairs + negative_pairs

    def load_image_tensor(self, path: str) -> torch.Tensor:
        """Load single image as tensor (for evaluation)."""
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image).unsqueeze(0)