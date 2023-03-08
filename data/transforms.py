from torchvision.transforms import transforms
from data.gaussian_blur import GaussianBlur
import data.HSI_transforms as HSI_transforms
import yaml

def get_simclr_data_transforms(input_shape, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=eval(input_shape)[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
                                          transforms.ToTensor()])
    return data_transforms


# def get_crophflip_HSI_transforms(input_shape):
#     data_transforms = HSI_transforms.Compose([
#                                               HSI_transforms.RandomResizedCrop(size=eval(input_shape)[-2:]),
#                                               HSI_transforms.RandomHorizontalFlip(p=0.5)
#     ])
#     return data_transforms
#
# def get_byol_HSI_all_transforms(input_shape, probs):
#     target_transforms = HSI_transforms.Compose([HSI_transforms.RandomResizedCrop(size=eval(input_shape)[-2:], scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
#                                               HSI_transforms.RandomHorizontalFlip(p=0.8*probs[0]),
#                                               HSI_transforms.RandomSubwindowHorizontalFlip(p=0.8*probs[1], ws=(1,3)),
#                                               HSI_transforms.RandomRotation(degrees=(0,270), p=0.8*probs[2]),
#                                               HSI_transforms.RandomPixelErasing(p=0.8*probs[3], scale=(1.0/81.0,16.0/81.0))
#     ])
#
#     online_transforms = HSI_transforms.Compose([HSI_transforms.RandomResizedCrop(size=eval(input_shape)[-2:], scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
#                                               HSI_transforms.RandomSubwindowVerticalFlip(p=0.2*probs[0], ws=(1,3)),
#                                               HSI_transforms.RandomVerticalFlip(p=0.2*probs[1]),
#                                               HSI_transforms.RandomSubwindowRotation(degrees=(0,270), ws=(1,3), p=0.2*probs[2]),
#                                               HSI_transforms.RandomPixelErasing(p=0.2*probs[3], scale=(1.0/81.0, 16.0/81.0))
#     ])
#
#     return target_transforms, online_transforms


def get_byol_HSI_single_transforms(input_shape, probs1, probs2):
    target_transforms = HSI_transforms.Compose([HSI_transforms.RandomResizedCrop(size=eval(input_shape)[-2:], scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
                                              HSI_transforms.RandomHorizontalFlip(p=1.0*probs1[0]),
                                              HSI_transforms.RandomSubwindowHorizontalFlip(p=1.0*probs1[1], ws=(1,3)),
                                              HSI_transforms.RandomRotation(degrees=(1, 359), p=1.0*probs1[2]),
                                              HSI_transforms.RandomPixelErasing(p=1.0*probs1[3], scale=(1.0/81.0,16.0/81.0))
    ])

    online_transforms = HSI_transforms.Compose([HSI_transforms.RandomResizedCrop(size=eval(input_shape)[-2:], scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
                                              HSI_transforms.RandomSubwindowVerticalFlip(p=1.0*probs2[0], ws=(1,3)),
                                              HSI_transforms.RandomVerticalFlip(p=1.0*probs2[1]),
                                              HSI_transforms.RandomSubwindowRotation(degrees=(1, 359), ws=(1,3), p=1.0*probs2[2]),
                                              HSI_transforms.RandomPixelErasing(p=1.0*probs2[3], scale=(1.0/81.0, 16.0/81.0))
    ])

    return target_transforms, online_transforms

# config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
# target_transforms = config['data_transforms']['target_transforms']
# online_transforms = config['data_transforms']['online_transforms']

# def get_individual_transform(target_transforms, online_transforms, input_shape, transform_name):
#     if transform_name==target_transforms[0]:
#         return HSI_transforms.RandomResizedCrop(size=eval(input_shape)[-2:], scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0))
#     elif transform_name==target_transforms[1]:
#         return HSI_transforms.RandomHorizontalFlip(p=1.0)
#     elif transform_name == target_transforms[2]:
#         return HSI_transforms.RandomSubwindowHorizontalFlip(p=1.0, ws=[1,3])
#     elif transform_name == target_transforms[3]:
#         return HSI_transforms.RandomRotation(degrees=(1, 90), p=1.0) #only support 90 180 270
#     elif transform_name == target_transforms[4]:
#         return HSI_transforms.RandomPixelErasing(p=1,  scale=(1.0/81.0, 25.0/81.0), ratio=(0.3, 3.3))
#     elif transform_name == target_transforms[5]:
#         return HSI_transforms.RandomBandErasing(p=1.0, scale=(0.05, 0.33))
#     elif transform_name == online_transforms[1]:
#         return HSI_transforms.RandomSubwindowVerticalFlip(p=1.0, ws=[1,3])
#     elif transform_name == online_transforms[2]:
#         return HSI_transforms.RandomVerticalFlip(p=1.0)
#     else:
#         return HSI_transforms.RandomSubwindowRotation(degrees=(1,90), ws=(1,3))