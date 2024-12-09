import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def prepare_model(num_classes):
    # Load pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Update the classifier for custom number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model_path = "model.pth"
num_classes = 2  # Update based on your dataset

# Prepare the model and load weights
model = prepare_model(num_classes)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)

# Set model to evaluation mode
model.eval()


# Define dummy input
dummy_input = torch.randn(1, 3, 224, 224)  # Replace with your model's expected input size

# Export the model
onnx_path = "model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,            # Store the trained parameter weights
    opset_version=11,              # Specify ONNX version
    input_names=['input'],         # Name for the input tensor
    output_names=['output'],       # Name for the output tensor
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic batching
)
print(f"Model successfully exported to {onnx_path}")