#!/usr/bin/env python3
"""Download CelebA files torchvision expects under <data>/celeba/."""

import os
import sys

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CELEBA_DIR = os.path.join(DATA_ROOT, "celeba")

FILES = [
    ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "img_align_celeba.zip"),
    ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "list_attr_celeba.txt"),
    ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "identity_CelebA.txt"),
    ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "list_bbox_celeba.txt"),
    ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "list_landmarks_align_celeba.txt"),
    ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "list_eval_partition.txt"),
]


def have_celeba():
    images = os.path.join(CELEBA_DIR, "img_align_celeba")
    attrs = os.path.join(CELEBA_DIR, "list_attr_celeba.txt")
    split = os.path.join(CELEBA_DIR, "list_eval_partition.txt")
    return os.path.isdir(images) and os.path.isfile(attrs) and os.path.isfile(split)


def try_torchvision():
    import torchvision.datasets as datasets

    print("trying torchvision CelebA download...", flush=True)
    datasets.CelebA(root=DATA_ROOT, split="train", download=True)
    return have_celeba()


def try_gdown():
    try:
        import gdown
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q gdown")
        import gdown

    os.makedirs(CELEBA_DIR, exist_ok=True)
    print("trying gdown CelebA files...", flush=True)
    for file_id, filename in FILES:
        dest = os.path.join(CELEBA_DIR, filename)
        if os.path.exists(dest) or (
            filename.endswith(".zip") and os.path.isdir(os.path.join(CELEBA_DIR, "img_align_celeba"))
        ):
            print("exists", filename, flush=True)
            continue
        url = f"https://drive.google.com/uc?id={file_id}"
        print("download", filename, flush=True)
        gdown.download(url, dest, quiet=False)
    zip_path = os.path.join(CELEBA_DIR, "img_align_celeba.zip")
    if os.path.isfile(zip_path) and not os.path.isdir(os.path.join(CELEBA_DIR, "img_align_celeba")):
        import zipfile

        print("extracting img_align_celeba.zip...", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(CELEBA_DIR)
    return have_celeba()


def main():
    if have_celeba():
        print("CelebA already present at", CELEBA_DIR)
        return 0
    os.makedirs(CELEBA_DIR, exist_ok=True)
    try:
        if try_torchvision():
            print("CelebA ready via torchvision")
            return 0
    except Exception as error:
        print("torchvision download failed:", error, flush=True)
    if try_gdown():
        print("CelebA ready via gdown")
        return 0
    print("CelebA download failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
