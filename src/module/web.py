import numpy as np
import cv2

import urllib.request

sample_image_urls = [
    'https://www.cha.go.kr/unisearch/images/national_treasure/2016012713433003.JPG',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Cheomseongdae-1.jpg/1200px-Cheomseongdae-1.jpg',
    'https://tourimage.interpark.com/BBS/Tour/FckUpload/201207/6347676482867224740.jpg',
    'https://file.mk.co.kr/meet/neds/2016/09/image_readtop_2016_661478_14743592592619280.jpg',
    'https://www.kbmaeil.com/news/photo/201905/816750_842857_5947.jpg']

def get(url):
    with urllib.request.urlopen(url) as xhr:
        return xhr.read()

def get_image(url):
    raw_frame = get(url)
    frame = np.asarray(bytearray(raw_frame), dtype=np.uint8)
    return cv2.imdecode(frame, cv2.IMREAD_COLOR)