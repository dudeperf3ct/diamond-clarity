import cv2
import albumentations as A


# strong augmentations
def get_train_transforms(height, width, means, stds):
    """
    Apply training transformations from albumentation library
    """
    trn_transform = A.Compose(
        [
            A.CenterCrop(height, width, cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(p=0.2),
            A.Perspective(p=0.5),
            A.Normalize(mean=means, std=stds)
        ]
    )
    return trn_transform


def get_val_transforms(height, width, means, stds):
    """
    Apply val transformations from albumentation library
    """
    val_transform = A.Compose(
        [
            A.CenterCrop(height, width, cv2.INTER_NEAREST),
            A.Normalize(mean=means, std=stds)
        ]
    )
    return val_transform
