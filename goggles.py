import cv2
import sys
from deepface import DeepFace
import os

# GStreamer 파이프라인 정의
gstreamer_pipeline = (
    'libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1, format=NV12 ! '
    'videoconvert ! appsink'
)

# OpenCV VideoCapture 객체 생성
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("카메라 열기 실패")
    sys.exit(0)

# OpenCV 창 이름 설정 (듀얼 디스플레이용)
window_name_1 = "Display 1"
window_name_2 = "Display 2"

# 첫 번째 모니터의 DISPLAY 환경 변수 설정
os.environ["DISPLAY"] = ":0"  # 첫 번째 모니터

# 첫 번째 창에 대한 설정 (전체화면)
cv2.namedWindow(window_name_1, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name_1, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # 전체화면 설정
cv2.moveWindow(window_name_1, 0, 0)  # 첫 번째 화면에 창 위치 지정

# 두 번째 모니터의 DISPLAY 환경 변수 설정
os.environ["DISPLAY"] = ":1"  # 두 번째 모니터

# 두 번째 창에 대한 설정 (일반 창 크기 설정)
cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name_2, 640, 480)  # 창 크기 지정 (일반 크기)
cv2.moveWindow(window_name_2, 800, 0)  # 두 번째 화면에 창 위치 지정

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    # 프레임 상하 반전
    frame = cv2.flip(frame, 0)

    # 프레임 좌우 반전
    frame = cv2.flip(frame, 1)

    # 프레임 크기를 800x480으로 조정
    frame = cv2.resize(frame, (800, 480))

    try:
        # DeepFace 감정 분석
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            detector_backend='ssd',  # SSD 백엔드 사용
            enforce_detection=False  # 얼굴 감지 필수
        )

        # 분석된 얼굴 결과 처리
        for result in results:
            # 얼굴 영역 정보
            x = result['region']['x']
            y = result['region']['y']
            w = result['region']['w']
            h = result['region']['h']

            # 얼굴 박스 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 가장 우세한 감정
            emotion = result['dominant_emotion']

            # 박스 위에 감정 텍스트 표시
            cv2.putText(
                frame,
                emotion,
                (x, y - 10),  # 얼굴 박스 위에 텍스트 출력
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    except Exception as e:
        print(f"Emotion analysis failed: {e}")

    # 영상 출력 (두 디스플레이에 동일한 프레임 출력)
    cv2.imshow(window_name_1, frame)  # 첫 번째 모니터에 전체화면으로 출력
    cv2.imshow(window_name_2, frame)  # 두 번째 모니터에 일반 크기로 출력

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
