# =============================================================================
# evaluate
# =============================================================================
"""
This file implements the evaluation pipeline for vehicle Re-ID.
main.py calls evaluate() after each epoch to compute Rank-1 and mAP
on the query / gallery splits.

No training happens here — gradients are disabled and dropout is off.

Evaluation steps:
  1. extract query embeddings : model.eval() + no_grad()
                                  forward pass on 1103 query images
                                  -> (1103, 128) L2-normalized embeddings

  2. extract gallery embeddings : same pipeline on 31238 gallery images
                                  -> (31238, 128) L2-normalized embeddings

  3. distance matrix : 1 - query_emb @ gallery_emb.T
                                  -> (1103, 31238) cosine distances
                                  valid because all embeddings are L2-normalized

  4. sort by distance : argsort ascending per query row
                                  -> ranked gallery indices per query

  5. per-query loop : reorder gallery labels into ranking order
                                  same vehicle + same camera     -> ignored
                                  same vehicle + different camera -> true positive

  6. Rank-1 : fraction of queries where top-1 is correct

  7. mAP : mean Average Precision over all 1103 queries
                                  AP_i = (1/R_i) sum_k P(k) * rel(k)
                                  mAP  = mean(AP_i)

evaluate() returns a dict consumed by main.py:
  {
    "rank1" : float  — fraction of correct top-1 retrievals  [0, 1]
    "mAP"   : float  — mean Average Precision over all queries [0, 1]
  }

Target: beat the 36.0% val mAP cross-entropy baseline of the 2021 winners.

See: engine/evaluate.md — full theory and metric justification
See: data/dataloader.py — get_query_dataloader, get_test_dataloader
See: https://www.aicitychallenge.org/2021-evaluation-system/
"""

import torch
from torch.utils.data import DataLoader


def evaluate(
    model: torch.nn.Module,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Extracts embeddings, computes the distance matrix and returns Rank-1 and mAP.

    Args:
        model : VehicleViT — eval mode and no_grad applied inside
        query_loader : sequential dataloader over image_query/
        gallery_loader : sequential dataloader over image_test/
        device : torch.device — tensors moved here before forward pass

    Returns:
        dict with rank1 and mAP, both floats in [0, 1]
    """

    model.eval()

    with torch.no_grad():

        # extract query embeddings, vehicle ids and camera ids
        query_embeddings = []
        query_vehicle_ids = []
        query_camera_ids = []

        for images, vehicle_ids, camera_ids in query_loader:
            images = images.to(device)
            query_embeddings.append(model(images).cpu())
            query_vehicle_ids.append(vehicle_ids)
            query_camera_ids.append(camera_ids)

        # same for gallery
        gallery_embeddings = []
        gallery_vehicle_ids = []
        gallery_camera_ids = []

        for images, vehicle_ids, camera_ids in gallery_loader:
            images = images.to(device)
            gallery_embeddings.append(model(images).cpu())
            gallery_vehicle_ids.append(vehicle_ids)
            gallery_camera_ids.append(camera_ids)

    query_embeddings = torch.cat(query_embeddings) # (1103, 128)
    query_vehicle_ids = torch.cat(query_vehicle_ids) # (1103,)
    query_camera_ids = torch.cat(query_camera_ids) # (1103,)
    gallery_embeddings = torch.cat(gallery_embeddings) # (31238, 128)
    gallery_vehicle_ids = torch.cat(gallery_vehicle_ids) # (31238,)
    gallery_camera_ids = torch.cat(gallery_camera_ids) # (31238,)

    # cosine distance matrix — valid because embeddings are L2-normalized
    dist_matrix = (1 - query_embeddings @ gallery_embeddings.T).to(device)

    # sort each row by ascending distance -> gallery indices ranked by similarity
    sorted_matrix = torch.argsort(dist_matrix, dim=1)

    rank1_list = []
    ap_list = []

    for i in range(sorted_matrix.shape[0]):

        query_vid = query_vehicle_ids[i]
        query_cid = query_camera_ids[i]

        # reorder gallery labels to match the ranking order
        sorted_vids = gallery_vehicle_ids[sorted_matrix[i]]
        sorted_cids = gallery_camera_ids[sorted_matrix[i]]

        # same vehicle + same camera -> ignored, same vehicle + diff camera -> true positive
        same_mask = (sorted_vids == query_vid) & (sorted_cids == query_cid)
        good_mask = (sorted_vids == query_vid) & (sorted_cids != query_cid)

        valid_positions = torch.where(~same_mask)[0]
        good_positions = torch.where(good_mask)[0]

        # rank-1: is the first non-ignored position a true positive?
        rank1_list.append(good_mask[valid_positions[0]].float())

        # AP: precision at each true positive position, then average
        if len(good_positions) == 0:
            ap_list.append(torch.tensor(0.0))
        else:
            num = torch.arange(1, len(good_positions) + 1).float()
            denom = good_positions.float() + 1
            ap_list.append((num / denom).mean())

    return {
        "rank1": torch.stack(rank1_list).mean().item(),
        "mAP": torch.stack(ap_list).mean().item(),
    }