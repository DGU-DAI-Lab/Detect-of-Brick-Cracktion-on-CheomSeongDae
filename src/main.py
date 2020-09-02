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

def mkdir(path):
    if not os.path.exists(path): os.mkdir(path)
    return path

dataset_path_format = '../asset/dataset/{class_id}/{file}'
class_path = {
    None: mkdir('../asset/dataset'),
    0: mkdir('../asset/dataset/0'),
    1: mkdir('../asset/dataset/1')
}

# --------------------------------

def create_dataset(frame):
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
    data_set = collections.defaultdict(list)
    for class_id in [0,1]:
        for file in glob.glob(f'{class_path[class_id]}/*.bmp'):
            data_set[class_id].append(cv2.imread(file))
    return data_set

# --------------------------------

def fit_model(data_set):
    model = load_model('model/built-in-model.h5')
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
    x_train = x_data_shuffled[:int(n_data*0.8)]
    y_train = y_data_shuffled[:int(n_data*0.8)]
    x_test = x_data_shuffled[int(n_data*0.8):]
    y_test = y_data_shuffled[int(n_data*0.8):]
    # Fit model
    model.fit(x_train, y_train, epochs=16)
    # Evaluate model
    model.evaluate(x_test, y_test)
    model.save('model/built-in-model-trained.h5')

# --------------------------------

def main():
    frame = get_image(sample_image_urls[0])
    # Load dataset
    data_set = load_dataset() # create_dataset(frame)
    # Load Model
    if not os.path.exists('model/built-in-model-trained.h5'):
        fit_model(data_set)
    model = load_model('model/built-in-model-trained.h5')
    # Apply Model
    with CV2_UI_ImageWindow('Apply Model') as window:
        window.image = frame
        while chr(cv2.waitKey(10) & 0xFF) != 'q':
            class_prob = model.predict(np.expand_dims(window.cursor_frame, axis=0))[0]
            predict = np.where(class_prob==max(class_prob))[0][0]
            window.cursor_color = [(0,0,255), (0,196,196)][predict] # 커서 색상을 0번 레이블이면 빨강, 1번은 노랑으로 설정
            window.update()
main()