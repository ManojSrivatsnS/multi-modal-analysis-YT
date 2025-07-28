import torch.nn as nn
import torchvision.models as models
import torch  # Ensure torch is imported for the example input
# Ensure torch is imported for the example input
# This import is necessary for the example input to work correctly
# ResNet50End2End model for end-to-end processing of video frames
# The model uses a pre-trained ResNet50 backbone, removes the classification head,
# and adds a global average pooling layer followed by a regression layer.

class ResNet50End2End(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(pretrained=True)
        base_model.fc = nn.Identity()  # remove classification head
        self.cnn = base_model
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Linear(2048, 1)

    def forward(self, x):  # x: [B, 960, 3, 128, 128]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # merge batch and time
        feats = self.cnn(x)         # shape: [B*T, 2048]
        feats = feats.view(B, T, -1).transpose(1, 2)  # [B, 2048, T]
        pooled = self.global_pool(feats).squeeze(-1)  # [B, 2048]
        out = self.regressor(pooled).squeeze(-1)      # [B]
        return out

def resnet50_end2end():
    return ResNet50End2End()

if __name__ == "__main__":
    model = resnet50_end2end()
    print(model)

    # Example input: batch size 2, 10 time steps, 3 channels, 128x128 images
    example_input = torch.randn(2, 10, 3, 128, 128)
    output = model(example_input)
    print(output.shape)  # Expected output shape: [2]
    print(output)
    # Output should be a tensor of shape [B] where B is the batch size
    # This will print the output tensor with the expected shape
    # and the values will be random due to the random input tensor. 


    