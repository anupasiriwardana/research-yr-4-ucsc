# from ultralytics import YOLO
# model = YOLO("yolov8n.pt")
# print(model.model.model[0])   # inspect the layer STEM_LAYER_INDEX["yolov8"] points to

# import torch
# x = torch.zeros(1, 3, 640, 640)
# out = model.model.model[0](x)
# print(out.shape)   # e.g. [1, C, 320, 320] -> stride = 640/320 = 2

from ultralytics import YOLO
import torch

model = YOLO("yolov8n.pt")
# x = torch.zeros(1, 3, 640, 640)

# with torch.no_grad():
#     x = model.model.model[0](x)   # stem -- output: [1, 16, 320, 320]
#     print("layer 0:", x.shape)

#     x = model.model.model[1](x)   # feed layer 0's OUTPUT in, not the raw image
#     print("layer 1:", x.shape)

x = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    for i in range(6):  # check the first few layers
        x = model.model.model[i](x)
        print(f"layer {i}: {model.model.model[i].__class__.__name__}, shape {tuple(x.shape)}")