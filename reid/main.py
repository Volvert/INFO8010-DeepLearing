# =============================================================================
# main.py
# =============================================================================
"""
Orchestrates the full training pipeline.
All configuration is read from config/tiny_vit.yaml — edit that file
to change any hyperparameter, path, or monitoring setting.

Run:
    python main.py
    python main.py --config config/tiny_vit.yaml

To disable the synthetic dataset: set synthetic_root to null in the YAML.
To resume training:               set resume to the checkpoint path in the YAML.

Evaluation modes (data.eval_mode in tiny_vit.yaml):
    local    : splits real train into train/query/gallery using make_train_eval_split()
               40 held-out identities never seen during training → mAP meaningful
    official : uses image_query/ and image_test/ from the dataset
               vehicleIDs absent (-1) → mAP not meaningful locally
"""

import os
import argparse
import yaml
import torch

from data.dataset import VehicleReIDDataset, MergedDataset, make_train_eval_split
from data.dataloader import get_train_dataloader, get_query_dataloader, get_test_dataloader
from data.data_transforms import get_train_transform, get_test_transform
from model.init_model import build_model
from losses.tripletloss import BatchHardTripletLoss
from engine.train import train_one_epoch
from engine.evaluate import evaluate
from utils.schedular import build_scheduler
from monitoring.logger import Logger
from monitoring.gradient_health import GradientHealthMonitor
from monitoring.triplet_health import TripletHealthMonitor
from monitoring.plot import generate_all_plots


def main(config_path: str = "config/tiny_vit.yaml") -> None:

    # ── config ────────────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data = cfg["data"]
    train = cfg["training"]
    mon = cfg["monitoring"]

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ndevice : {device}")

    # ── datasets ──────────────────────────────────────────────────────────────
    eval_mode = data.get("eval_mode", "local")

    if eval_mode == "local":
        # ── proper 3-way split — no overlap between train and eval ────────────
        # Load full real train once (no transform — make_train_eval_split applies them)
        full_real = VehicleReIDDataset(
            root = os.path.join(data["real_root"], "image_train"),
            label_xml = os.path.join(data["real_root"], "train_label.xml"),
            transform = None, # transforms applied inside make_train_eval_split
            id_offset = 0,
        )

        real_train_ds, query_dataset, gallery_dataset = make_train_eval_split(
            dataset = full_real,
            n_eval_ids = data["n_eval_ids"],
            seed = data["eval_seed"],
            train_transform = get_train_transform(),
            eval_transform = get_test_transform(),
        )

        print(f"Train  : {real_train_ds}")
        print(f"Query  : {query_dataset}")
        print(f"Gallery: {gallery_dataset}")

        # optionally merge synthetic on top of real train
        if data["synthetic_root"]:
            # synthetic uses real_train_ds max label as offset to avoid ID collisions
            max_real_id = max(real_train_ds.labels)
            synthetic_train = VehicleReIDDataset(
                root = os.path.join(data["synthetic_root"], "sys_image_train"),
                label_xml = os.path.join(data["synthetic_root"], "train_label.xml"),
                transform = get_train_transform(),
                id_offset = max_real_id,
            )
            train_dataset = MergedDataset(real_train_ds, synthetic_train)
            print(f"Merged : {train_dataset}")
        else:
            train_dataset = real_train_ds

    else:
        # ── official AIC21 splits — vehicleIDs absent → mAP not meaningful ───
        real_train = VehicleReIDDataset(
            root = os.path.join(data["real_root"], "image_train"),
            label_xml = os.path.join(data["real_root"], "train_label.xml"),
            transform = get_train_transform(),
            id_offset = 0,
        )

        if data["synthetic_root"]:
            synthetic_train = VehicleReIDDataset(
                root = os.path.join(data["synthetic_root"], "sys_image_train"),
                label_xml = os.path.join(data["synthetic_root"], "train_label.xml"),
                transform = get_train_transform(),
                id_offset = max(real_train.labels),
            )
            train_dataset = MergedDataset(real_train, synthetic_train)
            print(f"{train_dataset}")
        else:
            train_dataset = real_train
            print(f"{real_train}")

        query_dataset = VehicleReIDDataset(
            root = os.path.join(data["real_root"], "image_query"),
            label_xml = os.path.join(data["real_root"], "query_label.xml"),
            transform = get_test_transform(),
        )
        gallery_dataset = VehicleReIDDataset(
            root = os.path.join(data["real_root"], "image_test"),
            label_xml = os.path.join(data["real_root"], "test_label.xml"),
            transform = get_test_transform(),
        )

    # ── dataloaders ───────────────────────────────────────────────────────────
    train_loader = get_train_dataloader(train_dataset, data["P"], data["K"], data["num_workers"])
    
    query_loader = get_query_dataloader(query_dataset,   data["eval_batch_size"], data["num_workers"])
    
    gallery_loader = get_test_dataloader(gallery_dataset, data["eval_batch_size"], data["num_workers"])

    # ── model / optimizer / scheduler / loss ──────────────────────────────────
    model = build_model(cfg["model"], device, checkpoint_path=train["resume"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = train["lr"],
        weight_decay = train["weight_decay"],
        betas = tuple(train["betas"]),
    )
    scheduler = build_scheduler(optimizer, train["warmup_epochs"], train["epochs"])
    loss_fn = BatchHardTripletLoss(
        margin = train["margin"],
        lambda_unif = train["lambda_unif"],
    )

    # ── monitoring ────────────────────────────────────────────────────────────
    logger = Logger(mon["run_name"], train["epochs"], mon["runs_dir"], config=cfg)
    grad_monitor = GradientHealthMonitor(log_every_n_batches=mon["grad_every"])
    triplet_monitor = TripletHealthMonitor(log_every_n_batches=1)

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(train["epochs"]):

        logger.epoch_start()

        train_metrics = train_one_epoch(
            model = model,
            dataloader = train_loader,
            loss_fn = loss_fn,
            optimizer = optimizer,
            device = device,
            margin = train["margin"],
            grad_monitor = grad_monitor,
            triplet_monitor = triplet_monitor,
        )

        scheduler.step()

        run_eval = (epoch % mon["eval_every"] == 0) or (epoch == train["epochs"] - 1)
        eval_metrics = evaluate(model, query_loader, gallery_loader, device) if run_eval else {}

        logger.log_epoch(epoch, train_metrics, eval_metrics)

        if logger.is_best():
            torch.save(model.state_dict(), logger.best_ckpt_path)
            logger.log_message(f"best mAP {logger.best_mAP:.4f} → {logger.best_ckpt_path}")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            triplet_monitor.report(train_metrics, epoch=epoch, margin=train["margin"])
            grad_monitor.report(train_metrics)

    # ── post-training ─────────────────────────────────────────────────────────
    generate_all_plots(run_dir=logger.run_dir, margin=train["margin"])
    logger.summary()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/tiny_vit.yaml")
    args = p.parse_args()
    main(config_path=args.config)