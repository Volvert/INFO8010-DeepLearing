# Code Architecture

## Data Initialization

```mermaid
flowchart TD
    XML("train_label.xml")
    T("data_transforms.py")
    D("dataset.py")
    B("batch.py")
    DL("dataloader.py")

    XML -- "parsed once at construction" --> D
    T -- "get_train_transform()\nget_test_transform()" --> D
    D -- "dataset.labels" --> B
    D -- "dataset" --> DL
    B -- "PKSampler" --> DL

    style XML fill:#E1F5EE,stroke:#1D9E75,color:#085041
    style T fill:#EEF0FE,stroke:#7F77DD,color:#3C3489
    style D fill:#FFF4E5,stroke:#E8A020,color:#7A4500
    style B fill:#FFF4E5,stroke:#E8A020,color:#7A4500
    style DL fill:#E6F1FB,stroke:#378ADD,color:#0C447C
```

```
train_label.xml       ← source of labels (vehicleID, cameraID) for each image
      ↓
data_transforms.py    ← two transform pipelines passed to the Dataset constructor
                            train : crop · flip · jitter · blur · erase · normalize
      ↓                     test  : resize · normalize — deterministic, required for kNN

dataset.py            ← reads XML, loads images on the fly, returns (tensor, vid, cid)
                         self.samples[i] = (img_path, vehicle_id, camera_id)
      ↓                  self.labels[i]  = vehicle_id — only attribute consumed by PKSampler

batch.py              ← receives dataset.labels, groups indices by vehicle_id
                         samples P=16 identities × K=4 images per batch
      ↓                  guarantees 3 positives and 60 negatives per anchor for the triplet loss

dataloader.py         ← wraps dataset + PKSampler into a PyTorch DataLoader
                            train : drop_last=True  — incomplete batch breaks the triplet loss
                            query/test : shuffle=False — fixed order required for kNN
```

## Model Construction

## Train and Evaluate Process

## Test and Monitoring

### Losses

### Gradients
