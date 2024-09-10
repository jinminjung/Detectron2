
import cv2
import os


def get_files(path):
    for root, subdirs, files in os.walk(path):
     
        list_files = []

        if len(files) > 0:
            for f in files:
                fullpath = root + '/' + f
                list_files.append(fullpath)

    return list_files

def sort_key(file_path):
    base_name = os.path.basename(file_path)
    number_part = ''.join(filter(str.isdigit, base_name))
    return int(number_part)

image_files = get_files('./output')

image_files.sort(key=sort_key)

img = cv2.imread(image_files[0])
height,width,channel = img.shape # 이미지 크기를 가져옵니다.
fps = 5 # fps를 25로 합니다.

# 현재 소스코드가 있는 경로에 mp4 포맷으로 저장하도록 합니다.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))


for file in image_files:

    img = cv2.imread(file)

    writer.write(img)

    cv2.imshow("result", img)


    key = cv2.waitKey(int(1000/fps)) # fps를 사용하여 delay 설정

    if key == 27:     # ESC키 누르면 중지
        break

writer.release()
cv2.destroyAllWindows()