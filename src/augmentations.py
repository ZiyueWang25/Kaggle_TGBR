import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import util

class Base:
    img_scale = (1664, 936) #(1280, 720), (1408, 792), (1536, 864), (1664, 936)
    
    HorizontalFlip = 0.5
    Flip = 0
    RandomRotate90 = 0
    HueSaturationValue = 0
    RandomBrightnessContrast = 0
    RandomSizedBBoxSafeCrop = 0
    use_mosaic = False
    use_mixup = False

class Base5(Base):
    HorizontalFlip = 0.5
    Flip = 0.5
    RandomRotate90 = 0.5
    HueSaturationValue = 0.5
    RandomBrightnessContrast = 0.5
    RandomSizedBBoxSafeCrop = 0.5
    
class Base0(Base):
    HorizontalFlip = 0

class BaseMosaic(Base):
    use_mosaic = True

class BaseMixUp(Base):
    use_mixup = True
    
def read_hyp_param(name):
    assert name in globals(), "name is not in " + str(globals())
    hyp_param = globals()[name]
    hyp_param = util.class2dict(hyp_param)
    return hyp_param


def get_albu_transforms(aug_param, is_train=True):
    img_scale = aug_param['img_scale']
    aug_list = []
    if aug_param.get("HorizontalFlip", 0) > 0:
        aug_list.append(A.HorizontalFlip(p=aug_param["HorizontalFlip"]))
    if aug_param.get("Flip", 0) > 0:
        aug_list.append(A.HorizontalFlip(p=aug_param["Flip"]))
    if aug_param.get("HueSaturationValue", 0) > 0:
        aug_list.append(A.HueSaturationValue(p=aug_param["HueSaturationValue"])) 
    if aug_param.get("RandomBrightnessContrast", 0) > 0:
        aug_list.append(A.RandomBrightnessContrast(p=aug_param["RandomBrightnessContrast"])) 
    if aug_param.get("RandomSizedBBoxSafeCrop", 0) > 0:
        aug_list.append(
            A.RandomSizedBBoxSafeCrop(
                int(img_scale[0] * 0.8), 
                int(img_scale[1] * 0.8),
                p=aug_param["RandomSizedBBoxSafeCrop"])
        )
    aug_list.append(ToTensorV2(p=1))
    return convert_augs_to_mmdet_dict(A.Compose(aug_list, p=1.0))
    


def train_transforms(
    img_size=[512, 512],
    bbox_format="pascal_voc",
):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            # https://www.hertsdiveclub.co.uk/blog/colour-loss-divers-perspective/
            # Less red in deeper water
            # A.RGBShift(
            #     r_shift_limit=(-255, 0), g_shift_limit=(-50, 0), b_shift_limit=(0, 0)
            # ),
            A.Equalize(p=1),
            # A.CLAHE(p=1),
            A.OneOf(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=0.2,
                        sat_shift_limit=0.2,
                        val_shift_limit=0.2,
                        p=0.9,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.9
                    ),
                ],
                p=0.9,
            ),
            # A.ShiftScaleRotate(p=1),
            # A.OpticalDistortion(distort_limit=1.0, p=1),
            A.RandomSizedBBoxSafeCrop(int(img_size[0] * 0.8), int(img_size[1] * 0.8)),
            A.Resize(img_size[0], img_size[1], p=1),
            A.PadIfNeeded(img_size[0], img_size[1], p=1),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format=bbox_format, min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def valid_transforms(
    img_size=[512, 512],
    bbox_format="pascal_voc",
):
    return A.Compose(
        [
            A.Equalize(p=1),
            # A.CLAHE(p=1),
            A.Resize(img_size[0], img_size[1], p=1),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format=bbox_format, min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def _rename_keys(list_of_dicts, old="__class_fullname__", new="type"):
    for d in list_of_dicts:
        d[new] = d.pop(old)
        del d["id"]

        # The OneOf transforms need further unpacking
        if "transforms" in d.keys():
            d["transforms"] = _rename_keys(d["transforms"])
            del d["params"]

    return list_of_dicts


def convert_augs_to_mmdet_dict(augmentations):
    aug_list = augmentations.get_dict_with_id()["transforms"]

    # Drop ToTensorV2
    aug_list.pop(-1)

    aug_list = _rename_keys(aug_list)

    return aug_list


if __name__ == "__main__":
    augs = train_transforms()
    aug_list = convert_augs_to_mmdet_dict(augs)
    print(aug_list)
