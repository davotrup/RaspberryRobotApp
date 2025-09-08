from flask import Flask, Response, render_template, jsonify, request, send_file
from picamera2 import Picamera2
import cv2
import numpy as np
import os
import time
import io
import subprocess
import re
from gpiozero import Motor
from threading import Lock
import threading
import pickle

# ====== МОТОРЫ ======
motor_left = Motor(forward=22, backward=23)
motor_right = Motor(forward=17, backward=27)

# Прибавка скорости (0.0–0.3), накапливается из UI
speed = 0.0
speed_lock = Lock()

# ====== APP ======
app = Flask(__name__)
start_time = time.time()

# ====== КАМЕРА ======
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# ====== РАСПОЗНАВАНИЕ ЛИЦ ======
# Функция для поиска или скачивания каскада Хаара
def find_haarcascade_file(filename):
    # Проверяем в текущей директории
    if os.path.exists(filename):
        return filename

    # Проверяем стандартные пути
    possible_paths = [
        '/usr/share/opencv4/haarcascades/',
        '/usr/local/share/opencv4/haarcascades/',
        '/usr/share/opencv/haarcascades/',
        '/usr/share/opencv/'
    ]

    for path in possible_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path

    # Если не нашли, пробуем скачать
    try:
        print(f"Скачиваем {filename}...")
        import urllib.request
        url = f'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{filename}'
        urllib.request.urlretrieve(url, filename)
        return filename
    except Exception as e:
        print(f"Не удалось скачать {filename}: {e}")
        return None

# Загружаем каскад
cascade_path = find_haarcascade_file('haarcascade_frontalface_default.xml')
if cascade_path:
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print("Каскад Хаара загружен успешно")
else:
    face_cascade = None
    print("Внимание: каскад Хаара не найден, распознавание лиц отключено")

# ----- ПАРАМЕТРЫ ПРИЗНАКОВ -----
# Поддерживаем прежние имена переменных: EXPECTED_FEATURE_SIZE используется при валидации.
# Для LBP-гистограммы размер признака = 256 бинов.
FACE_FEATURE_SIZE = 16  # значение не используется напрямую, оставлено для обратной совместимости
EXPECTED_FEATURE_SIZE = 256

# Порог для принятия решения (хи-квадрат расстояние для LBP-гистограмм)
CHI2_THRESHOLD = 1.9  # при необходимости подстрой: 0.5..1.0

# ----- ПРЕДОБРАБОТКА + LBP -----
def _clahe_gray(img_gray):
    # CLAHE выравнивает освещенность — критично для ИК/ночной камеры
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img_gray)

def _elliptical_mask(h, w, scale=0.9):
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * scale / 2), int(h * scale / 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask

def _lbp_histogram(gray_128):
    """Быстрый LBP (8-соседей) с векторизацией + нормированная гистограмма на 256 бинов."""
    # берём внутреннюю область без рамки 1 пиксель
    g = gray_128.astype(np.uint8)

    # Сдвиги (y, x)
    shifts = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    weights = [128, 64, 32, 16, 8, 4, 2, 1]

    # центральная часть
    c = g[1:-1, 1:-1]
    codes = np.zeros_like(c, dtype=np.uint16)

    for (dy, dx), w in zip(shifts, weights):
        nb = g[1+dy: -1+dy if -1+dy != 0 else None, 1+dx: -1+dx if -1+dx != 0 else None]
        # сравнение соседей с центром
        codes |= ((nb > c).astype(np.uint16) * w)

    hist, _ = np.histogram(codes.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)  # L1-нормализация
    return hist

# Простая функция для извлечения признаков лица (заменена на LBP-гистограмму)
def extract_face_features(face_image):
    try:
        if face_image is None or face_image.size == 0:
            print("Пустое изображение лица")
            return None

        # Приводим к оттенкам серого
        if len(face_image.shape) == 3:
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_image

        # Выравнивание яркости/контраста → лучше для ИК-съёмки
        gray_face = _clahe_gray(gray_face)

        # Приводим к стандартному размеру (больше деталей, чем 64×64)
        resized = cv2.resize(gray_face, (128, 128), interpolation=cv2.INTER_AREA)

        # Мягкая фильтрация от шума (оставим очень лёгкую, чтобы не смазать текстуру)
        resized = cv2.GaussianBlur(resized, (3, 3), 0)

        # Маска эллипсом, чтобы вырезать фон вокруг лица
        mask = _elliptical_mask(128, 128, scale=0.9)
        resized = cv2.bitwise_and(resized, resized, mask=mask)

        # Извлекаем LBP-гистограмму (256-мерный вектор)
        feature_vector = _lbp_histogram(resized)

        # Валидация размера
        if feature_vector.shape[0] != EXPECTED_FEATURE_SIZE:
            print(f"Ошибка: неверный размер вектора признаков: {feature_vector.shape[0]}, ожидалось: {EXPECTED_FEATURE_SIZE}")
            return None

        return feature_vector
    except Exception as e:
        print(f"Ошибка извлечения признаков: {e}")
        return None

# Загрузка/сохранение известных лиц
known_faces = {}  # dict: name -> np.ndarray(256,)
face_embeddings_file = 'known_faces.pkl'

def load_known_faces():
    """Загружаем усреднённые признаки из known_faces.pkl."""
    global known_faces
    try:
        if os.path.exists(face_embeddings_file):
            with open(face_embeddings_file, 'rb') as f:
                loaded = pickle.load(f)

            # Фильтруем по корректному размеру
            valid = {}
            for name, vec in loaded.items():
                if vec is not None and isinstance(vec, np.ndarray) and vec.shape[0] == EXPECTED_FEATURE_SIZE:
                    valid[name] = vec.astype(np.float32)
                else:
                    # Поддержка старых форматов (списки и т.п.)
                    try:
                        arr = np.array(vec, dtype=np.float32).reshape(-1)
                        if arr.shape[0] == EXPECTED_FEATURE_SIZE:
                            valid[name] = arr
                        else:
                            print(f"Пропускаю {name}: неверный размер ({arr.shape[0]})")
                    except Exception:
                        print(f"Пропускаю {name}: невозможно преобразовать вектор признаков")

            known_faces = valid
            print(f"Загружено {len(known_faces)} известных лиц")
        else:
            known_faces = {}
            print("Файл с известными лицами не найден")
    except Exception as e:
        print(f"Ошибка загрузки известных лиц: {e}")
        known_faces = {}

def save_known_faces():
    try:
        valid = {}
        for name, vec in known_faces.items():
            if vec is not None and isinstance(vec, np.ndarray) and vec.shape[0] == EXPECTED_FEATURE_SIZE:
                valid[name] = vec.astype(np.float32)
        with open(face_embeddings_file, 'wb') as f:
            pickle.dump(valid, f)
        print(f"Сохранено {len(valid)} известных лиц")
    except Exception as e:
        print(f"Ошибка сохранения известных лиц: {e}")

def _chi2_distance(h1, h2):
    """Хи-квадрат расстояние между нормированными гистограммами."""
    # 0.5 * sum( (a-b)^2 / (a+b) )
    num = (h1 - h2) ** 2
    den = (h1 + h2) + 1e-8
    return 0.5 * np.sum(num / den, dtype=np.float32)

def train_known_faces():
    """Обучение на нескольких изображениях одного человека: усреднение LBP-гистограмм."""
    global known_faces

    if face_cascade is None:
        print("Каскад Хаара не загружен, обучение невозможно")
        return 0

    known_faces = {}
    trained_count = 0

    faces_dir = './faces'
    if not os.path.exists(faces_dir):
        print("Папка faces не найдена")
        return 0

    # Временное накопление: имя -> список векторов
    temp_faces = {}

    for filename in os.listdir(faces_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Имя берём из начала до "_image"
        name = filename.split("_image")[0].strip()
        if not name:
            print(f"Пропуск файла без имени: {filename}")
            continue

        path = os.path.join(faces_dir, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"Ошибка чтения {filename}")
            continue

        # Детектируем лицо на исходном изображении
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
        )
        if len(faces) == 0:
            print(f"Лицо не найдено на {filename}")
            continue

        # Берём наиболее крупное лицо (чаще всего правильное)
        (x, y, w, h) = max(faces, key=lambda r: r[2]*r[3])
        face_roi = img[y:y+h, x:x+w]

        features = extract_face_features(face_roi)
        if features is not None and features.shape[0] == EXPECTED_FEATURE_SIZE:
            temp_faces.setdefault(name, []).append(features)
            print(f"Добавлено фото для {name}")
        else:
            print(f"Ошибка признаков для {filename}")

    # Усредняем признаки по каждому человеку
    for name, feats in temp_faces.items():
        if len(feats) > 0:
            avg = np.mean(np.stack(feats, axis=0), axis=0).astype(np.float32)
            # Допнормализация на случай численных перекосов
            s = float(avg.sum())
            if s > 1e-8:
                avg = avg / s
            known_faces[name] = avg
            trained_count += 1
            print(f"Сохранено лицо: {name} (на основе {len(feats)} фото)")

    save_known_faces()
    return trained_count

def recognize_face(face_roi):
    """Сравнение LBP-гистограмм по χ²-расстоянию."""
    try:
        if len(known_faces) == 0:
            return "Unknown (no trained faces)"

        current = extract_face_features(face_roi)
        if current is None or current.shape[0] != EXPECTED_FEATURE_SIZE:
            return "Unknown"

        best_name = "Unknown"
        best_dist = 1e9

        for name, ref in known_faces.items():
            dists = [_chi2_distance(current, v) for v in ref]
            d = min(dists)
            if d < best_dist:
                best_dist = d
                best_name = name

        if best_dist < CHI2_THRESHOLD:
            return best_name
        else:
            return f"Unknown ({best_dist:.2f})"
    except Exception as e:
        print(f"Ошибка распознавания: {e}")
        return "Unknown"

# ====== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ======
frame_lock = threading.Lock()
current_frame = None

# ====== SLAM ======
MAP_SIZE = 600
map_lock = Lock()
last_map_png = None

def make_blank_map():
    img = np.ones((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8) * 245
    for i in range(0, MAP_SIZE, 50):
        cv2.line(img, (i, 0), (i, MAP_SIZE), (220, 220, 220), 1)
        cv2.line(img, (0, i), (MAP_SIZE, i), (220, 220, 220), 1)
    cv2.circle(img, (MAP_SIZE//2, MAP_SIZE//2), 8, (0, 0, 255), -1)
    cv2.putText(img, 'ROBOT', (MAP_SIZE//2 + 10, MAP_SIZE//2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return img

# ====== ТЕЛЕМЕТРИЯ ======
def get_wifi_rssi():
    try:
        out = subprocess.check_output(["iwconfig", "wlan0"], text=True, stderr=subprocess.STDOUT)
        m = re.search(r"Signal level=(-?\d+) dBm", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read())/1000.0, 1)
    except Exception:
        return None

def get_uptime():
    return int(time.time() - start_time)

# ====== ВИДЕО-ПОТОК ======
def generate_frames():
    global current_frame
    while True:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.1)
                continue

            # Преобразуем 4-канальное изображение в 3-канальное
            if current_frame.shape[2] == 4:
                frame = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2BGR)
            else:
                frame = current_frame.copy()

        # Обнаружение лиц (если каскад загружен)
        if face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Распознавание и отрисовка лиц
            for (x, y, w, h) in faces:
                # Вырезаем область лица
                face_roi = frame[y:y+h, x:x+w]

                # Распознаем лицо
                name = recognize_face(face_roi)

                # Рисуем прямоугольник и подпись
                color = (0, 255, 0) if name.startswith('Daniil') or name.startswith('Aleksandra') else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            # Если каскад не загружен, просто показываем сообщение
            cv2.putText(frame, "Face detection disabled", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Кодируем кадр в JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

# ====== ПОТОК КАМЕРЫ ======
def camera_thread():
    global current_frame
    while True:
        try:
            frame = picam2.capture_array()
            with frame_lock:
                current_frame = frame
        except Exception as e:
            print("Camera error:", e)
            time.sleep(0.1)

# ====== SLAM ПОТОК ======
def slam_thread():
    global last_map_png
    map_img = make_blank_map()

    while True:
        with map_lock:
            _, enc = cv2.imencode('.png', map_img)
            last_map_png = enc.tobytes()
        time.sleep(0.1)

# ====== МАРШРУТЫ ======
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/slam_map')
def slam_map_feed():
    def gen():
        while True:
            with map_lock:
                if last_map_png is None:
                    time.sleep(0.05)
                    continue
                img_bytes = last_map_png
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + img_bytes + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/map.png')
def map_png():
    with map_lock:
        if last_map_png is None:
            img = make_blank_map()
            _, enc = cv2.imencode('.png', img)
            return Response(enc.tobytes(), mimetype='image/png')
        return Response(last_map_png, mimetype='image/png')

@app.route('/telemetry')
def telemetry():
    with speed_lock:
        s = speed
    data = {
        'speed': round(s, 2),
        'wifi_dbm': get_wifi_rssi(),
        'cpu_temp_c': get_cpu_temp(),
        'uptime_s': get_uptime(),
    }
    return jsonify(data)

@app.route('/battery')
def battery():
    """Возвращаем уровень батареи в процентах"""
    return jsonify(level=100.0)  # Заглушка

@app.route('/train_faces')
def train_faces():
    count = train_known_faces()
    return jsonify(status="ok", trained_faces=count)

@app.route('/reset_faces')
def reset_faces():
    """Сброс обученных лиц"""
    global known_faces
    known_faces = {}
    if os.path.exists(face_embeddings_file):
        os.remove(face_embeddings_file)
    return jsonify(status="ok", message="Обученные лица сброшены")

@app.route("/set_speed")
def set_speed():
    global speed
    try:
        value = float(request.args.get("value", 0.0))
        speed = max(0.0, min(0.3, value))
        return jsonify(status="ok", speed=speed)
    except:
        return jsonify(status="error"), 400

@app.route("/forward")
def forward():
    motor_left.forward(0.7 + speed)
    motor_right.forward(0.63 + speed)
    return jsonify(status="ok")

@app.route("/backward")
def backward():
    motor_left.backward(0.5 + speed)
    motor_right.backward(0.43 + speed)
    return jsonify(status="ok")

@app.route("/left")
def left():
    motor_left.backward(0.67 + speed)
    motor_right.forward(0.6 + speed)
    return jsonify(status="ok")

@app.route("/right")
def right():
    motor_left.forward(0.5 + speed)
    motor_right.backward(0.5 + speed)
    return jsonify(status="ok")

@app.route("/stop")
def stop():
    motor_left.stop()
    motor_right.stop()
    return jsonify(status="ok")

if __name__ == '__main__':
    # Загружаем известные лица
    load_known_faces()

    # Если нет обученных лиц, пробуем обучить
    if len(known_faces) == 0:
        print("Попытка обучения на известных лицах...")
        train_known_faces()

    print(f"Готово! Загружено {len(known_faces)} известных лиц")

    # Запускаем потоки
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=slam_thread, daemon=True).start()

    print("Сервер запускается на http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
