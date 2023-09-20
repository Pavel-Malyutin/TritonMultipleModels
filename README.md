It is a simple repository wich can compare work speed of two methots of model inference on CPU. The first one is Triton Inference Server and the second one is native with FastAPI.

This program contains three models to recognise registration plates.

At first you need to run
```
docker-compose up -d
```
to build and run servers.

Than you need to install requirements. Afret it you can run client applications fastapi_client.py and pytriton_client.py to test the models.

# NN
Package with neural models for recognition of russian car license plates. Model were taken from [this repository](https://github.com/EtokonE/License_Plate_Recognition).
Detection pipeline:
- License plate detection with [YoloV5](https://github.com/ultralytics/yolov5)
- License plate alignment with [STN](https://pytorch.org/tutorials//intermediate/spatial_transformer_tutorial.html)
- Text recognition with [LPR-Net](https://www.sciencedirect.com/science/article/abs/pii/S0167865518306998)


