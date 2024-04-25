import torch
import torch.nn as nn
import torchvision
import yaml
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class LogicityFeatureExtractor(nn.Module):
    def __init__(self):
        super(LogicityFeatureExtractor, self).__init__()
        
        # Build feature extractor
        self.resnet_layer_num = 4
        self.resnet_fpn = resnet_fpn_backbone(
            "resnet50", pretrained=True, trainable_layers=self.resnet_layer_num+1
        )
        self.img_feature_channels = self.resnet_fpn.out_channels * self.resnet_layer_num # 1024
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, batch_imgs, batch_bboxes):
        B = batch_imgs.shape[0]
        N = batch_bboxes.shape[1]
        batch_imgs = batch_imgs.permute(0, 3, 2, 1) # Bx3xHxW
        # normalize with mean and std of ImageNet
        batch_imgs = self.transform(batch_imgs)
        
        # Extract img features
        with torch.no_grad(): # frozen
            fpn_features = self.resnet_fpn(batch_imgs)
        feature_list = []
        for layer in range(self.resnet_layer_num):
            feature = fpn_features[str(layer)]
            feature = torch.nn.functional.interpolate(feature, fpn_features["0"].shape[-2:], mode="bilinear")
            feature_list.append(feature)
        imgs_features = torch.cat(feature_list, dim=1)
        imgs_features = torch.nn.functional.interpolate(imgs_features, batch_imgs.shape[-2:], mode="bilinear") # B x C_img x H x W
        
        # Get bbox features
        bboxes = [batch_bboxes[i] for i in range(batch_bboxes.shape[0])]
        roi_features = torchvision.ops.roi_pool(imgs_features, bboxes, output_size=1).squeeze()
        roi_features = roi_features.view(B, N, roi_features.shape[1]) # B x N x C_img

        return roi_features