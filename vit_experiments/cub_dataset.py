import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def resolve_cub_root(path):
    """Accept either .../cub-200 or .../cub-200/CUB_200_2011."""
    root = Path(path).expanduser().resolve()
    if (root / "images.txt").is_file():
        return root
    candidate = root / "CUB_200_2011"
    if (candidate / "images.txt").is_file():
        return candidate
    raise FileNotFoundError(
        f"Cannot find CUB metadata under {root}. Expected images.txt either "
        "there or in a CUB_200_2011 subdirectory."
    )


def _read_int_map(path, subtract_one=False):
    result = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.strip().split(maxsplit=1)
            value = int(value)
            result[int(key)] = value - 1 if subtract_one else value
    return result


class CUB200(Dataset):
    """Official CUB split with optional precomputed per-image text embeddings."""

    def __init__(
        self,
        root,
        train,
        transform,
        semantic_root=None,
        semantic_key="embedding_words",
        semantic_dim=768,
        max_semantic_tokens=32,
    ):
        self.root = resolve_cub_root(root)
        self.transform = transform
        self.semantic_root = None if semantic_root is None else Path(semantic_root).expanduser().resolve()
        self.semantic_key = semantic_key
        self.semantic_dim = semantic_dim
        self.max_semantic_tokens = max_semantic_tokens

        labels = _read_int_map(self.root / "image_class_labels.txt", subtract_one=True)
        splits = _read_int_map(self.root / "train_test_split.txt")
        expected_split = 1 if train else 0
        self.samples = []
        with open(self.root / "images.txt", "r", encoding="utf-8") as handle:
            for line in handle:
                image_id, relative_path = line.strip().split(maxsplit=1)
                image_id = int(image_id)
                if splits[image_id] == expected_split:
                    self.samples.append((relative_path, labels[image_id]))

        with open(self.root / "classes.txt", "r", encoding="utf-8") as handle:
            self.classes = [line.strip().split(maxsplit=1)[1] for line in handle]

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in official split under {self.root}")
        if self.semantic_root is not None and not self.semantic_root.is_dir():
            raise FileNotFoundError(f"Semantic root does not exist: {self.semantic_root}")

    def __len__(self):
        return len(self.samples)

    def _load_semantics(self, relative_path):
        if self.semantic_root is None:
            return torch.empty(0, self.semantic_dim, dtype=torch.float32)

        semantic_path = self.semantic_root / Path(relative_path).with_suffix(".pickle")
        if not semantic_path.is_file():
            raise FileNotFoundError(f"Missing semantic feature: {semantic_path}")
        with open(semantic_path, "rb") as handle:
            payload = pickle.load(handle)
        value = payload[self.semantic_key] if isinstance(payload, dict) else payload
        value = np.asarray(value, dtype=np.float32)
        value = np.squeeze(value)
        if value.ndim == 1:
            value = value[None, :]
        if value.ndim != 2 or value.shape[-1] != self.semantic_dim:
            raise ValueError(
                f"Expected [tokens, {self.semantic_dim}] in {semantic_path}, got {value.shape}"
            )
        value = value[: self.max_semantic_tokens]
        # The generator has no padding mask. Fill unused slots with the sample's
        # mean semantic vector instead of introducing artificial zero tokens.
        output = np.repeat(
            value.mean(axis=0, keepdims=True), self.max_semantic_tokens, axis=0
        ).astype(np.float32, copy=False)
        output[: len(value)] = value
        return torch.from_numpy(output)

    def __getitem__(self, index):
        relative_path, target = self.samples[index]
        image_path = self.root / "images" / relative_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        semantics = self._load_semantics(relative_path)
        return image, target, semantics
