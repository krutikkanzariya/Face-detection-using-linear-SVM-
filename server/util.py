import joblib
import json
import numpy as np
import cv2
import base64
from wavelet import w2d

def classify_image(image_base64_data,file_path:None):
    imgs=get_cropped_image_if_2_eyes(file_path,image_base64_data)
    result=[]
    for img in imgs:
        scalled_row_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_row_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        len_image_array =32*32*3 + 32*32;
        final = combined_img.reshape(1,len_image_array).astype(float)

        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.round(__model.predict_proba(final)*100).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result


__model = None
__class_name_to_number = {}
__class_number_to_name = {}

def load_artifacts():
    print("loading artifacts...")
    global __class_name_to_number
    global  __class_number_to_name
    with open("./artifacts/class_dictionary.json",'r') as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}
    global __model
    if __model is None:
        with open("./artifacts/saved_model.pkl","rb") as f:
            __model = joblib.load(f)

    print("artifacts are loded.......")

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]


def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #print(len(faces))
    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)

    return cropped_faces


def get_b64_test_image_for_virat():
    with open("b64.txt") as f:
        return f.read()


if __name__== '__main__':
    load_artifacts()
    #print(classify_image(get_b64_test_image_for_virat(),None))
    #print(classify_image(None,"./test_image/virat1.jpg"))
    #print(classify_image(None,"./test_image/maria1.jpg"))
    #print(classify_image(None,"./test_image/saraben1.jpg"))
    #print(classify_image(None,"./test_image/feral1.jpg"))
    print(classify_image(None, "./test_image/vir11.jpg"))
    #print(class_number_to_name(5))