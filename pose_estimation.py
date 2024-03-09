import cv2
from ultralytics import YOLO
import numpy as np
import cvzone

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)  


# Çıktı için codec'i tanımladık  ve bir VideoWriter nesnesi oluşturduk.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
output_width = 800
output_height = 450
out = cv2.VideoWriter('posees.mp4', fourcc, 30, (output_width, output_height)) 

# Üç anahtar nokta arasındaki açıyı hesaplamak için bir fonksiyon tanımla
def calculate_angle(a, b, c):
    a = np.array(a)  # İlk
    b = np.array(b)  # Orta
    c = np.array(c)  # Son
    
# Radyan cinsinden açıyı hesaplar ve dereceye çevirir.
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
# Açı 180 dereceden büyükse, 360'dan çıkararak düzeltir.

    if angle > 180.0:
        angle = 360 - angle

    return angle

while True:
    ret, frame = cap.read()
    
    # Frame'i yeniden boyutlandır, başarısız olursa geç
    try:
        frame = cv2.resize(frame, (output_width, output_height))
    except:
        pass

    # Daha fazla frame kalmadığında döngüyü kır
    if not ret:
        break

    # Model ile tahmin yap
    results = model.predict(frame, save=True)

    # Sınırlayıcı kutu bilgilerini xyxy formatında al
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    statuses = []

    # Algılanan tüm kişiler için anahtar nokta verilerini al
    keypoints_data = results[0].keypoints.data

    # Algılanan kişileri yinele
    for i, keypoints in enumerate(keypoints_data):
        # Anahtar noktaların algılandığından emin ol
        if keypoints.shape[0] > 0:
            
            #11 numara kafayı,13 kalçayı,15 dizi temsil eder
    
            angle = calculate_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])  # Kafa, kalça ve dizler arasındaki açıları aldık
            print(f"Kişi {i + 1} {'Sitting' if angle is not None and angle < 110 else 'Standing'} (Açı: {angle:.2f} derece)")
            statuses.append('Sitting' if angle is not None and angle < 110 else 'Standing')

    # Frame üzerine sınırlayıcı kutuları ve durumları çiz
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(
            frame, f"{statuses[i]}", (x1, y2 - 10),
            scale=3, thickness=3,
            colorT=(255, 255, 255), colorR=(255, 0, 255),
            font=cv2.FONT_HERSHEY_PLAIN,
            offset=10,
            border=0, colorB=(0, 255, 0)
        )

    # Frame'i çıktı video dosyasına yaz
    out.write(frame)
    detection = results[0].plot()
    cv2.imshow('YOLOv8 Pose Detection', detection)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
