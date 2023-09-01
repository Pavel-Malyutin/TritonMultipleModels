# NN
Package with neural models for recognition of russian car license plates. Model were taken from [this repository](https://github.com/EtokonE/License_Plate_Recognition).
Detection pipeline:
- License plate detection with [YoloV5](https://github.com/ultralytics/yolov5)
- License plate alignment with [STN](https://pytorch.org/tutorials//intermediate/spatial_transformer_tutorial.html)
- Text recognition with [LPR-Net](https://www.sciencedirect.com/science/article/abs/pii/S0167865518306998)


