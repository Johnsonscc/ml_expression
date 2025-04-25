import torch
from ultralytics import YOLO
import os

def train_model():
    # 设置CUDA环境
    device = torch.device("cuda")
    # model = YOLO('last01.pt')  # 加载网络结构
    model = YOLO('yolov8l-p6.yaml').load('yolov8l.pt')
    # 进行模型训练
    model.train(
        # 从头开始训练
        data='data.yaml',
        epochs=200,
        imgsz=640,
        device=device,
        workers =0,
        batch =16,
    )
    # 进行模型验证
    model.val()
if __name__ == "__main__":
    # 调用训练函数
    train_model()

    #  python yolo_train.py
