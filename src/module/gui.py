import cv2
import numpy as np

class CV2_UI_ImageWindow:
    def __init__(self, winname=None):
        self.winname = winname if winname else str(id(self))
        self._image = None
        # mouse & cursor 관련 설정
        self.mx = 0 # mouse x pos
        self.my = 0 # mouse y pos
        self.cursor_width = 32
        self.cursor_height = 32
        self.cursor_border_size = 1
        self.cursor_color = (0,0,255)
        self.cursor_frame = None
        # 이미지 출력 전 작업
        self.process = lambda x: x
        # 창 생성
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, self.onmouse)
        self.update()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        cv2.destroyWindow(self.winname)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, new_image):
        self._image = new_image
        self.update()

    @property
    def cursor_points(self):
        height,width = self.image.shape[:2]
        if (self.cursor_width > width) or (self.cursor_height > height):
            raise Exception('box size must be smaller than image size')
        x0 = min(max(0, self.mx - self.cursor_width//2), width - self.cursor_width)
        y0 = min(max(0, self.my - self.cursor_height//2), height - self.cursor_height)
        x1 = x0 + self.cursor_width
        y1 = y0 + self.cursor_height
        return (x0, y0), (x1, y1)

    def onmouse(self,evt,x,y,*args):
        if evt == cv2.EVENT_MOUSEMOVE:
            self.mx = x
            self.my = y
            self.update()
    
    def update(self):
        if self.image is None:
            self.image = np.zeros((64,64,3), dtype=np.uint8)
        else:
            frame = self.image.copy()
            # Make cursor
            (cx0, cy0), (cx1, cy1) = self.cursor_points
            self.cursor_frame = self.image[cy0:cy1, cx0:cx1]
            cv2.rectangle(frame, (cx0, cy0), (cx1-1, cy1-1), self.cursor_color, self.cursor_border_size)
            cv2.imshow(self.winname, frame)
            # Do something
            self.process(self)

if __name__ == '__main__':
    # 사용 예시
    # 1. 이미지 불러오기
    from web import get_image, sample_image_urls
    img_url = sample_image_urls[0] 
    sample_img = get_image(img_url)
    # 2. CV2_UI_ImageWindow 생성하여 사용
    with CV2_UI_ImageWindow('Example') as win: # with문 내부에 머무는 동안 창이 유지됨.
        win.image = sample_img # 이미지 지정
        while (cv2.waitKey(10) & 0xFF) != ord('q'): # 'q'버튼을 누르기 전까지는 창이 닫히지 않음.
            cv2.imshow(f'{win.winname}-cursor', win.cursor_frame) # 커서가 기리키는 영역은 '.cursor_frame'으로 접근하여 사용