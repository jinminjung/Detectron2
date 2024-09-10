import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import common libraries
import numpy as np
import os, cv2, time
import torch
import json 

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

# Load bonnet coordinates from JSON file
with open("/home/elicer/hkheon/service/video2_bonnet_xy.json", "r") as f:
    bonnet_coords = json.load(f)["bonnet_coordinates"]

# Convert bonnet coordinates to a NumPy array for masking
bonnet_polygon = np.array(bonnet_coords, dtype=np.int32)

# Register custom dataset with a different name
DatasetCatalog.register("my_custom_dataset", lambda: get_custom_dataset_dicts("/home/elicer/dataset_sort/merged_labels_yolo/10020000.json"))
MetadataCatalog.get("my_custom_dataset").thing_classes = ["Parking Space", "Driveable Space", "car", "person"]

# 메타데이터를 초기화하고 색상 설정
metadata = MetadataCatalog.get("my_custom_dataset")
metadata.thing_colors = [fixed_colors[cls] for cls in metadata.thing_classes]

predictor = DefaultPredictor(cfg)

# Define paths for input and output videos
video_path = "/home/elicer/detectron2_custom_dataset/video/inference/video3.mp4"
output_video_path = "/home/elicer/detectron2_custom_dataset/video/service/result/video3_service_16_final.mp4"

# Distance calculation parameters
H_camera = 120  # 카메라의 실제 높이(cm)
focal_length = 500  # 카메라의 초점 거리(픽셀)
distance_threshold = 4  # 4미터 이내의 경우 경고 표시

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

# Define bonnet area to exclude
bonnet_threshold_y = int(height * 0.8)  # 하단 20%를 본넷 영역으로 설정

# 비디오 프로세싱 메인 루프
while cap.isOpened():
    start_time = time.time()  # 시작 시간 기록

    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break  # 더 이상 읽을 프레임이 없으면 루프 탈출

    # 본넷 영역을 제외한 마스크 적용
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    mask[:bonnet_threshold_y, :] = 255  # 상단 80%만 활성화

    # 현재 프레임에서 객체 탐지 수행
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)  # 본넷 부분을 제외한 프레임으로 예측 수행
    outputs = predictor(masked_frame)
    instances = outputs["instances"].to("cpu")

    # 주차공간 탐지를 위한 필터링
    parking_spaces = instances[instances.pred_classes == metadata.thing_classes.index("Parking Space")]

    # 신뢰도가 90 이상인 "Parking Space"만 필터링
    parking_spaces = parking_spaces[parking_spaces.scores > 0.9]
    num_parking_spaces = len(parking_spaces)

    # 결과 시각화 설정
    out_frame = frame.copy()

    for i in range(len(instances)):
        # 신뢰도 계산
        confidence = instances.scores[i].item() * 100  # 신뢰도 퍼센트로 변환

        # 특정 신뢰도 이상만 출력
        if confidence >= 90:
            mask = instances.pred_masks[i].numpy()
            cls_id = instances.pred_classes[i].item()
            label_text = metadata.thing_classes[cls_id]
            color = fixed_colors[label_text]

            # 마스크 색상으로 채우기
            colored_mask = np.zeros_like(out_frame)
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]

            # 본넷 좌표에 해당하는 부분을 제외한 마스크 적용
            cv2.fillPoly(colored_mask, [bonnet_polygon], (0, 0, 0))  # 본넷 좌표에 해당하는 부분은 검정색으로 덮어씀

            # 남아 있는 마스크 부분을 영상에 오버레이
            out_frame = cv2.addWeighted(out_frame, 1.0, colored_mask, 0.5, 0)

            # 본넷 영역과 겹치지 않는 부분만 남긴 마스크 생성
            mask_remained = mask.copy().astype(np.uint8)
            cv2.fillPoly(mask_remained, [bonnet_polygon], 0)  # 본넷 영역을 0으로 설정

            # 남아 있는 마스크를 기반으로 테두리 그리기
            contours, _ = cv2.findContours(mask_remained, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out_frame, contours, -1, color, 2)  # 테두리 추가

            # 라벨명과 신뢰도를 표시
            bbox = instances.pred_boxes[i].tensor.numpy().astype(int)[0]
            x, y = bbox[0], bbox[1]  # 좌상단 좌표
            text = f"{label_text}: {confidence:.1f}%"
            cv2.putText(out_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)  # 글씨 크기 줄임

    # FPS 계산
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0

    # FPS 및 주차 공간 수를 프레임에 표시
    cv2.putText(out_frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(out_frame, f'Parking Spaces: {num_parking_spaces}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 신뢰도 90 이상인 주차공간 위에 화살표 그리기
    for box in parking_spaces.pred_boxes:
        x_center = int((box[0] + box[2]) / 2)
        y_top = int(box[1])

        arrow_start = (x_center, y_top - 10)
        arrow_end = (x_center, y_top + 20)

        cv2.arrowedLine(out_frame, arrow_start, arrow_end, (0, 255, 0), 2, tipLength=0.5)

    # 사람에 대한 거리 계산 및 시각화
    persons = instances[instances.pred_classes == metadata.thing_classes.index("person")]

    for i in range(len(persons)):
        bbox = persons.pred_boxes[i].tensor.numpy()[0]
        x1, y1, x2, y2 = bbox

        # y축 거리 계산 (사람의 바운딩 박스 높이)
        person_height_in_pixels = abs(y2 - y1)

        # 거리 계산
        distance = (H_camera * focal_length) / person_height_in_pixels
        distance_m = distance / 100  # cm를 m로 변환

        # 거리 정보 표시 제거
        xmid = (x1 + x2) / 2
        cv2.putText(out_frame, f'{distance_m:.2f} m', (int(xmid), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 4미터 이내인 경우 큰 느낌표 표시
        if distance_m <= distance_threshold:
            # 사람 중심에 동그라미 배경을 갖는 느낌표 그리기
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            radius = 30
            cv2.circle(out_frame, (center_x, center_y), radius, (255, 255, 255), -1)  # 흰색 배경

            # 느낌표 크기 조정 및 가운데 맞추기
            text_size = cv2.getTextSize('!', cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2

            # 느낌표 표시
            cv2.putText(out_frame, '!', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # 수정된 프레임을 출력 비디오 파일에 저장
    out.write(out_frame)

cap.release()
out.release()
print("Video processing complete and resources released.")