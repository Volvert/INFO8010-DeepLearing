from data.dataset import VehicleReIDDataset
from data.dataloader import get_train_dataloader, get_query_dataloader, get_test_dataloader
from data.data_transforms import get_train_transform, get_test_transform

train_dataset = VehicleReIDDataset(
    root="dataset/AIC21_Track2_ReID/image_train",
    label_xml="dataset/AIC21_Track2_ReID/train_label.xml",
    transform=get_train_transform()
)
query_dataset = VehicleReIDDataset(
    root="dataset/AIC21_Track2_ReID/image_query",
    label_xml="dataset/AIC21_Track2_ReID/query_label.xml",
    transform=get_test_transform()
)
test_dataset = VehicleReIDDataset(
    root="dataset/AIC21_Track2_ReID/image_test",
    label_xml="dataset/AIC21_Track2_ReID/test_label.xml",
    transform=get_test_transform()
)