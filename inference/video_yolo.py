import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import common libraries
import numpy as np
import os, cv2, time
import torch

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

# Define paths and parameters
config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
num_classes = 4  
device = "cuda"  # Or "cpu"

# Load configuration and create predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
cfg.MODEL.WEIGHTS = os.path.join("/home/elicer/detectron2_custom_dataset/output_yolo3/segmentation/model_final.pth")  # Path to your trained model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Set the number of classes (including background)
cfg.MODEL.DEVICE = device
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Custom threshold

# 고정된 색상 설정
fixed_colors = {
    "Parking Space": (0, 255, 255),  # Yellow for Parking Space
    "Driveable Space": (0, 255, 0),  # Green for Driveable Space
    "car": (255, 0, 0),  # Blue for Car
    "person": (0, 0, 255)  # Red for Person
}

# Register custom dataset with a different name
DatasetCatalog.register("my_custom_dataset", lambda: get_custom_dataset_dicts("/home/elicer/dataset_sort/merged_labels_yolo/10020000.json"))
MetadataCatalog.get("my_custom_dataset").thing_classes = ["Parking Space", "Driveable Space", "car", "person"]

# 메타데이터를 초기화하고 색상 설정
metadata = MetadataCatalog.get("my_custom_dataset")
metadata.thing_colors = [fixed_colors[cls] for cls in metadata.thing_classes]

predictor = DefaultPredictor(cfg)

# Define paths for input and output videos
video_path = "/home/elicer/detectron2_custom_dataset/video/inference/video3.mp4"
output_video_path = "/home/elicer/detectron2_custom_dataset/video/inference/result/720p/video3_output_server_yolo16.mp4"

# 비디오 캡처 객체 초기화
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# 비디오 코덱 및 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 비디오 프로세싱 메인 루프
while cap.isOpened():
    start_time = time.time()  # 시작 시간 기록

    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break  # 더 이상 읽을 프레임이 없으면 루프 탈출

    # 현재 프레임에서 객체 탐지 수행
    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")

    # 결과 시각화
    out_frame = frame.copy()

    # 모든 인스턴스를 시각화
    for i in range(len(instances)):
        cls_id = instances.pred_classes[i].item()
        label_text = metadata.thing_classes[cls_id]

        # 마스크 가져오기
        mask = instances.pred_masks[i].numpy()

        # 통일된 색상 설정
        mask_color = fixed_colors[label_text]

        # 마스크를 통일된 색상으로 채우기
        colored_mask = np.zeros_like(out_frame)
        for c in range(3):
            colored_mask[:, :, c] = mask * mask_color[c]

        # 마스크를 영상에 오버레이
        out_frame = cv2.addWeighted(out_frame, 1.0, colored_mask, 0.5, 0)
        
        # 마스크 테두리 그리기
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_frame, contours, -1, mask_color, 2)  # 테두리 추가

        # 라벨명과 신뢰도 가져오기
        confidence = instances.scores[i].item() * 100  # confidence를 퍼센트로 변환
        text = f"{label_text}: {confidence:.1f}%"

        # 바운딩 박스 좌표 가져오기
        bbox = instances.pred_boxes[i].tensor.numpy().astype(int)[0]
        x, y = bbox[0], bbox[1]  # 좌상단 좌표

        # 라벨명과 신뢰도를 마스크 색상으로 영상에 표시
        cv2.putText(out_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mask_color, 2)

    # FPS 계산
    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    
    # FPS를 프레임에 표시
    cv2.putText(out_frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 수정된 프레임을 출력 비디오 파일에 저장
    out.write(out_frame)

# 모든 리소스 해제
cap.release()
out.release()
print("Video processing complete and resources released.")
