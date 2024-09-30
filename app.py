import os
import pathlib
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import cv2  # OpenCV 추가

# 경로 수정 (Windows 환경 대응)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 프로젝트 설정
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['RESULT_FOLDER'] = './static/results'  # 결과 이미지 저장 폴더 추가
app.config['SECRET_KEY'] = 'supersecretkey'

# 결과 이미지 저장 폴더가 없으면 생성
if not os.path.exists(app.config['RESULT_FOLDER']):
    os.makedirs(app.config['RESULT_FOLDER'])

# YOLO 모델 로드 (best.pt 파일 경로 설정)
MODEL_PATH = './model/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

# 파일 업로드 허용 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 확장자 검사 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 이미지 리사이즈 함수
def resize_image(image_path, output_size=(640, 480)):
    # 이미지 읽기
    image = cv2.imread(image_path)
    # 이미지 리사이즈
    resized_image = cv2.resize(image, output_size)
    # 리사이즈된 이미지 저장
    resized_image_path = os.path.join(app.config['RESULT_FOLDER'], 'resized_' + os.path.basename(image_path))
    cv2.imwrite(resized_image_path, resized_image)
    return resized_image_path

# 메인 페이지 (이미지 업로드 폼)
@app.route('/')
def index():
    return render_template('index.html')

# 이미지 업로드 및 철도 이상 감지 처리
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # YOLO 모델로 철도 이상 감지 처리
        img = Image.open(filepath)
        results = model(img)

        # 결과 이미지 저장
        results_dir = './runs/detect/exp'  # YOLO가 결과를 저장하는 기본 경로
        results.save()  # YOLO가 기본적으로 결과를 저장합니다.
        
        # 가장 최근에 저장된 이미지 찾기
        result_image_path = os.path.join(results_dir, filename)

        # 리사이즈된 이미지 저장
        resized_image_path = resize_image(result_image_path)

        return redirect(url_for('uploaded_file', filename=os.path.basename(resized_image_path)))
    
    flash('Allowed file types are png, jpg, jpeg')
    return redirect(request.url)

# 결과 페이지 (감지된 이미지 출력)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    # 서버 실행
    app.run(debug=True) 