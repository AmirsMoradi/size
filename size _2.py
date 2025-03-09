import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

scale = 200
cx, cy = 500, 500
showImage = False

img1 = cv2.imread("kiki.jpg")
if img1 is None:
    print("خطا: تصویر مورد نظر یافت نشد.")
    exit()

h1, w1, _ = img1.shape

while True:
    success, img = cap.read()
    if not success:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        showImage = True

        if len(results.multi_hand_landmarks) == 2:
            hand1 = results.multi_hand_landmarks[0]
            hand2 = results.multi_hand_landmarks[1]

            x1, y1 = int(hand1.landmark[mp_hands.HandLandmark.WRIST].x * img.shape[1]), int(hand1.landmark[mp_hands.HandLandmark.WRIST].y * img.shape[0])
            x2, y2 = int(hand2.landmark[mp_hands.HandLandmark.WRIST].x * img.shape[1]), int(hand2.landmark[mp_hands.HandLandmark.WRIST].y * img.shape[0])

            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            scale = int(length)  # اندازه تصویر متناسب با فاصله دست‌ها
            scale = max(50, min(1000, scale))  # محدودیت برای اندازه تصویر

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    else:
        showImage = False

    if showImage:
        newH, newW = scale, scale
        img1_resized = cv2.resize(img1, (newW, newH))


        y1, y2 = max(0, cy - newH // 2), min(720, cy + newH // 2)
        x1, x2 = max(0, cx - newW // 2), min(1280, cx + newW // 2)

        if y2 - y1 > 0 and x2 - x1 > 0:
            img[y1:y2, x1:x2] = img1_resized[:y2 - y1, :x2 - x1]

    # نمایش فریم
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()