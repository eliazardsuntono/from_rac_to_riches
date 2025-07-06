import cv2
import math
import cvzone
from ultralytics import YOLO

model = YOLO('weights/best.pt')
labels = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

# Loading the image
img1 = cv2.imread('media/garbage_1.jpg')
result1 = model(img1)

for r in result1:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1

        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        if conf > 0.3:
            cvzone.cornerRect(img1, (x1, y1, w, h), t=2)
            cvzone.putTextRect(img1, f'{labels[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1,
                               colorR=(255, 0, 0))

cv2.imshow("Image", img1)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cv2.waitKey(1)