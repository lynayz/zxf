# flexible-yolov5





*Update the code for  [ultralytics/yolov5](https://github.com/ultralytics/yolov5) version 6.1.*


## Prerequisites

please refer requirements.txt

## Getting Started

### Dataset Preparation

Make data for yolov5 format. you can use od/data/transform_voc.py convert VOC data to yolov5 data format.

### Training and Testing

For training and Testing, it's same like yolov5.

#### Training

1. check out configs/data.yaml, and replace with your data， and number of object nc
2. check out configs/model_*.yaml, choose backbone. and change nc to your dataset. please refer support_backbone in models.backbone.__init__.py
3. 
```shell script
$ python scripts/train.py  --batch 16 --epochs 5 --data configs/data.yaml --cfg configs/model_XXX.yaml
```

```shell
# for nvidia tensor-core 4:2 sparsity, install apex

git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

```

A google colab demo in train_demo.ipynb

#### Testing and Visualize

```shell script
$ python scripts/eval.py   --data configs/data.yaml  --weights runs/train/yolo/weights/best.py
```

### Detection

```shell
python scripts/detector.py   --weights yolov5.pth --imgs_root  test_imgs   --save_dir  ./results --img_size  640  --conf_thresh 0.4  --iou_thresh 0.4
```

### Deploy

#### Export 
```shell
python scripts/export.py   --weights yolov5.pth 
```

#### Grpc Server

In projects folder, tf_serving and triton demo are provided. 

#### Quantization

You can directly quantify the onnx model

```shell
python scripts/trt_quant/generate_int8_engine.py --onnx path --images-dir  img_path  --save-engine  engine_path
```
[See](scripts/trt_quant/README)


#### Tensorrt Inference

For tensorrt model, you can direct use official trt export, and refer scripts/trt_infer/cpp/. For test, I use TensorRT-8.4.0.6.

privode c++ / python demo, scripts/trt_infer

## &#8627; Contributors
<a href="https://github.com/Bobo-y/flexible-yolov5/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Bobo-y/flexible-yolov5" />
</a>
   
## &#8627; Stargazers
[![Stargazers repo roster for @Bobo-y/flexible-yolov5](https://reporoster.com/stars/Bobo-y/flexible-yolov5)](https://github.com/Bobo-y/flexible-yolov5/stargazers)

## &#8627; Forkers
[![Forkers repo roster for @Bobo-y/flexible-yolov5](https://reporoster.com/forks/Bobo-y/flexible-yolov5)](https://github.com/Bobo-y/flexible-yolov5/network/members)
