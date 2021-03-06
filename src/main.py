import numpy as np
import cv2

import functools
import time
import os
import glob
import collections

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from module.gui import CV2_UI_ImageWindow
from module.web import get_image, sample_image_urls
from core.determine_possibility import sequence

# --------------------------------

ENABLE_CREATING_DATASET = False # 이 값을 True로 변경하면, 데이터 셋을 생성하는 과정이 활성화 된다.

# --------------------------------

def mkdir(path):
    if not os.path.exists(path): os.mkdir(path)
    return path

dataset_path_format = '../asset/dataset/{class_id}/{file}'
class_path = {
    None: mkdir('../asset/dataset'),
    0: mkdir('../asset/dataset/0'),
    1: mkdir('../asset/dataset/1')}

# --------------------------------

def create_dataset(frame):
    # 주어진 이미지 (frame) 으로 부터 직접 데이터 셋을 생성.
    # * 커서가 가리키는 영역을 추출하는 방법은 키보드 버튼 '0' 혹은 '1'을 누르면 된다.
    #   눌린 버튼 숫자와 동일한 이름의 폴더아래 이미지가 저장된다.
    #   (샘플 데이터셋은 "0: 정상 영역 / 1: 균열 영역" 으로 분류한 예시이다.)ㄴ
    # * UI는 키보드 버튼 'Q'를 누르면 종료된다.
    data_set = collections.defaultdict(list)
    # Create Dataset
    with CV2_UI_ImageWindow('Create Dataset') as window:
        mask = functools.reduce(lambda x,func:func(x), sequence, frame) # frame에 대하여 sequence의 함수를 순차적으로 적용
        window.image = cv2.addWeighted(frame, 0.65, mask, 0.8, 0) # 창에는 마스크 되어진 이미지를 출력
        while True:
            # Cursor
            (x0, y0), (x1, y1) = window.cursor_points
            cursor_frame = frame[y0:y1, x0:x1]
            cv2.imshow(f'{window.winname}-cursor', cursor_frame)
            # Select Options
            key = chr(cv2.waitKey(10) & 0xFF)
            if key == 'q': break
            elif key in '01':
                class_id = int(key)
                # Save classify data (as dict)
                data_set[class_id].append(cursor_frame)
                # Save classify data (as local file)
                file = os.path.join(class_path[class_id], f'{time.time()}.bmp')
                cv2.imwrite(file, cursor_frame) # 원래 이미지에서 선택된 영역을 추출
                print(f'Image saved as {file}')
    return data_set

def load_dataset():
    # 'asset/dataset'으로 부터 사전에 생성된 데이터 셋을 불러옴.
    data_set = collections.defaultdict(list)
    for class_id in [0,1]:
        for file in glob.glob(f'{class_path[class_id]}/*.bmp'):
            data_set[class_id].append(cv2.imread(file))
    return data_set

# --------------------------------

def fit_model(data_set):
    TRAIN_DATA_RATIO = 0.8 # 전체 데이터 중 train 데이터로 사용 될 데이터의 비율 (나머지는 test 데이터로 사용)
    try:
        model = load_model('model/built-in-model.h5')
    except Exception:
        from model.built_in_model import model
    # Make dataset
    x_data = np.array(data_set[0] + data_set[1]) / 255.0
    y_data = to_categorical([0]*len(data_set[0]) + [1]*len(data_set[1]))
    n_data = len(x_data)
    # Shuffle dataset
    shuffled_indices = np.arange(n_data)
    np.random.shuffle(shuffled_indices)
    x_data_shuffled = x_data[shuffled_indices]
    y_data_shuffled = y_data[shuffled_indices]
    # Seperate dataset
    n_train_data = int(n_data * TRAIN_DATA_RATIO)
    x_train = x_data_shuffled[:n_train_data]
    y_train = y_data_shuffled[:n_train_data]
    x_test = x_data_shuffled[n_train_data:]
    y_test = y_data_shuffled[n_train_data:]
    # Fit model
    model.fit(x_train, y_train, epochs=16)
    # Evaluate model
    model.evaluate(x_test, y_test)
    model.save('model/built-in-model-trained.h5')
    return model

# --------------------------------

def main():
    frame = get_image(sample_image_urls[0]) # 모델에 사용할 이미지. 현재는 'src/module/web.py'의 get_image(url) 함수로 인터넷 이미지를 다운받아 사용한다.
    # Load dataset
    if ENABLE_CREATING_DATASET:
        data_set = create_dataset(frame)
    else:
        data_set = load_dataset()
    # Load Model
    try:
        model = load_model('model/built-in-model-trained.h5')
    except Exception:
        model = fit_model(data_set)
    # Apply Model
    with CV2_UI_ImageWindow('Apply Model') as window:
        window.image = frame
        while chr(cv2.waitKey(10) & 0xFF) != 'q':
            class_prob = model.predict(np.expand_dims(window.cursor_frame, axis=0))[0]
            predict = np.where(class_prob==max(class_prob))[0][0]
            window.cursor_color = [(0,0,255), (0,196,196)][predict] # 커서 색상을 0번 레이블이면 빨강, 1번은 노랑으로 설정
            window.update()
main()