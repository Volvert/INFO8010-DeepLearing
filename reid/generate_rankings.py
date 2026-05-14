"""
generate_rankings.py
--------------------
Uses a trained VehicleViT to rank gallery images for each query,
then writes one .txt file per query — the input format expected by visualize.py.

Usage:
    python generate_rankings.py --checkpoint runs/run_007/best_model.pth
    python generate_rankings.py --checkpoint runs/run_007/best_model.pth --top_k 50

Output:
    runs/run_007/rankings/
        000001.txt   ← ranked gallery IDs for query 000001
        000002.txt
        ...

Then run:
    python visualize.py
    → Txt Dir: runs/run_007/rankings/
"""

import os
import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset         import VehicleReIDDataset
from data.data_transforms import get_test_transform
from data.dataloader      import get_query_dataloader, get_test_dataloader
from model.init_model     import build_model


def extract_embeddings(model, loader, device):
    """Forward pass on a full split — returns embeddings and image stems."""
    model.eval()
    embeddings, img_stems = [], []

    with torch.no_grad():
        for images, _, _ in loader:
            embeddings.append(model(images.to(device)).cpu())

    # recover image filenames from the dataset
    for img_path, _, _ in loader.dataset.samples:
        img_stems.append(Path(img_path).stem)   # "000001" from "image_query/000001.jpg"

    return torch.cat(embeddings), img_stems


def generate_rankings(checkpoint_path: str, config_path: str, top_k: int = 50):

    # ── config + device ───────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data      = cfg["data"]
    real_root = data["real_root"]

    print(f"\ndevice     : {device}")
    print(f"checkpoint : {checkpoint_path}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg["model"], device, checkpoint_path=checkpoint_path)
    print(f"model      : loaded ✓")

    # ── datasets — always use official splits for visualize.py ────────────────
    query_dataset = VehicleReIDDataset(
        root      = os.path.join(real_root, "image_query"),
        label_xml = os.path.join(real_root, "query_label.xml"),
        transform = get_test_transform(),
    )
    gallery_dataset = VehicleReIDDataset(
        root      = os.path.join(real_root, "image_test"),
        label_xml = os.path.join(real_root, "test_label.xml"),
        transform = get_test_transform(),
    )

    print(f"query      : {len(query_dataset)} images")
    print(f"gallery    : {len(gallery_dataset)} images")

    # ── dataloaders ───────────────────────────────────────────────────────────
    query_loader   = get_query_dataloader(query_dataset,   batch_size=128, num_workers=4)
    gallery_loader = get_test_dataloader(gallery_dataset,  batch_size=128, num_workers=4)

    # ── extract embeddings ────────────────────────────────────────────────────
    print("\nExtracting query embeddings   ...")
    q_emb,  q_stems  = extract_embeddings(model, query_loader,   device)

    print("Extracting gallery embeddings ...")
    g_emb,  g_stems  = extract_embeddings(model, gallery_loader, device)

    # ── distance matrix ───────────────────────────────────────────────────────
    print("Computing distance matrix     ...")
    dist_matrix     = 1 - q_emb @ g_emb.T               # (N_query, N_gallery)
    ranked_indices  = torch.argsort(dist_matrix, dim=1)  # ascending distance

    # ── write txt files ───────────────────────────────────────────────────────
    run_dir      = Path(checkpoint_path).parent
    rankings_dir = run_dir / "rankings"
    rankings_dir.mkdir(exist_ok=True)

    print(f"Writing {len(q_stems)} ranking files → {rankings_dir}/\n")

    for i, q_stem in enumerate(q_stems):
        ranked_gallery_stems = [g_stems[j] for j in ranked_indices[i, :top_k].tolist()]
        txt_path = rankings_dir / f"{q_stem}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(ranked_gallery_stems))

    print(f"Done — {len(q_stems)} files written to {rankings_dir}/")
    print(f"\nNow run:")
    print(f"  python visualize.py")
    print(f"  → Txt Dir: {rankings_dir}/\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True,
                   help="path to best_model.pth  e.g. runs/run_007/best_model.pth")
    p.add_argument("--config",      default="config/tiny_vit.yaml")
    p.add_argument("--top_k",       type=int, default=50,
                   help="number of gallery images to rank per query (default 50)")
    args = p.parse_args()

    generate_rankings(
        checkpoint_path = args.checkpoint,
        config_path     = args.config,
        top_k           = args.top_k,
    )