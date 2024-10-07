from img_xtend.nn.autobackend import AutoBackend
from img_xtend.data.build import load_inference_source


model = AutoBackend(
    # weights="models/object_detection/yolov8m.pt",
    # weights="img_xtend/models/ultralytics/yolov8m.pt",
    weights="img_xtend/models/recognition/resnet.onnx",
)
model.warmup()

source = "Jake"
a = load_inference_source(source)
for i,elem in enumerate(a):
    model(elem[1][0])
    print(type(elem))
    print(elem)
    if i == 10:
        break