import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
# from torchvision.models.detection.rpn import AnchorGenerator, FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, AnchorGenerator
from torchvision.models.detection import FasterRCNN

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

__all__ = ["create_model"]


def create_model(num_classes, min_size=300, max_size=500, backbone="mobile_net"):
    # note num_classes = total_classes + 1 for background.

    # Adding multiple backbones We don't need the built in Fasterrcnn
    # This is the default backbone rcnn. We can change it.

    # This model was trained on COCO
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = model.to(device)

    # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features

    # ft_min_size = min_size
    # ft_max_size = max_size

    # These backbones are trained on ImageNet not on COCO
    # Please train them on COCO and provide model_dict I would use them instead.
    if backbone == "mobile_net":
        mobile_net = torchvision.models.mobilenet_v2(pretrained=True)
        # print(mobile_net.features) # From that I got the output channels for mobilenet
        ft_backbone = mobile_net.features
        ft_backbone.out_channels = 1280

    elif backbone == "vgg_11":
        vgg_net = torchvision.models.vgg11(pretrained=True)
        ft_backbone = vgg_net.features
        ft_backbone.out_channels = 512

    elif backbone == "vgg_13":
        vgg_net = torchvision.models.vgg13(pretrained=True)
        ft_backbone = vgg_net.features
        ft_backbone.out_channels = 512

    elif backbone == "vgg_16":
        vgg_net = torchvision.models.vgg16(pretrained=True)
        ft_backbone = vgg_net.features
        ft_backbone.out_channels = 512

    elif backbone == "vgg_19":
        vgg_net = torchvision.models.vgg19(pretrained=True)
        ft_backbone = vgg_net.features
        ft_backbone.out_channels = 512

    elif backbone == "resnet_18":
        resnet_net = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 512

    elif backbone == "resnet_34":
        resnet_net = torchvision.models.resnet34(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 512

    elif backbone == "resnet_50":
        resnet_net = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 2048

    elif backbone == "resnet_101":
        resnet_net = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 2048

    elif backbone == "resnet_152":
        resnet_net = torchvision.models.resnet152(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 2048

    elif backbone == "resnext101_32x8d":
        resnet_net = torchvision.models.resnext101_32x8d(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 2048
        # print(ft_model)

    else:
        print("Error Wrong unsupported Backbone")
        return

    ft_mean = [0.485, 0.456, 0.406]
    ft_std = [0.229, 0.224, 0.225]

    # ft_anchor_generator = AnchorGenerator(
    #     sizes=((32, 64, 128)), aspect_ratios=((0.5, 1.0, 2.0))
    # )

    ft_anchor_generator = AnchorGenerator(sizes=((2,4,8,16,32, 64, 128, 256,512),),
                                          aspect_ratios=((0.5, 1.0, 1.5, 2.0),))

    ft_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=[0], output_size=7, sampling_ratio=2
    )

    box_head = BoxHead(vgg)
    in_features = box_head.in_features

    ft_model = FasterRCNN(
        backbone=ft_backbone,
        num_classes=num_classes,
        # min_size=ft_min_size,
        # max_size=ft_max_size,
        rpn_anchor_generator=ft_anchor_generator,
        box_roi_pool=ft_roi_pooler,
        box_head = box_head,
        box_predictor = FastRCNNPredictor(in_features, num_classes)
    )
    # rpn_anchor_generator=ft_anchor_generator,
    # box_roi_pool=ft_roi_pooler)

    return ft_model


class BoxHead(nn.Module):
    def __init__(self, vgg):
        super(BoxHead, self).__init__()
        self.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        self.in_features = 4096 # feature out from mlp
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x
