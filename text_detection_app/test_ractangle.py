import cv2
import pytesseract
import tempfile
from PIL import Image
import numpy as np
import re
cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

#text 정제처리
def clean_text(read_data):
    text = re.sub("""[=+,#/\?:^$.@*\"※~&%ㆍ!』\\'|\(\)\[\]\<\>`\'…》£"¢¥"Ÿ«éȮϽñٶϴ»「—©]""", '', read_data)
    return text


def set_image_dpi(file_path):
    im = Image.open(file_path)
    print(im.size)
    length_x , length_y = im.size
    factor = max(1, int(1000/length_x))
    size = length_x*factor , length_y*factor
    im_resized = im.resize(size , Image.LANCZOS)
    temp_file = tempfile.NamedTemporaryFile(delete=False , suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(320,320))
    print(temp_filename)
    return temp_filename


def remove_noise_smooth(file_name):
    img = cv2.imread(file_name ,flags=0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255 , cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 41, 3)
    kernal = np.ones((1,1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel=kernal)
    closing = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel=kernal)
    ret1 , th1 = cv2.threshold(img , 127, 255, cv2.THRESH_BINARY)
    ret2 , th2 = cv2.threshold(th1, 0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
    img = cv2.GaussianBlur(th2, (1,1), 0)
    or_image = cv2.bitwise_or(img, closing)

    # print(or_image)
    # cv2.imshow("or_image",or_image)
    # cv2.waitKey(0)
    return or_image

def convert_gray_color(file_path):
    img = cv2.imread(file_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray_image)
    # cv2.imshow("이미지보이기"  , gray_image)
    # cv2.waitKey(0)
    return gray_image

def rectangle_detect(file_path, lang='kor'):
    gray_img = convert_gray_color(file_path)
    faces = cascade.detectMultiScale(gray_img , scaleFactor=1.1, minNeighbors=5 , minSize=(5,5))
    # print(faces)
    for b in faces:
        x , y ,w ,h = b
        img = cv2.rectangle(gray_img, (x,y) , (x+w , y+h) , (0,0,255) ,-1)
    # cv2.imshow("rectangle_face",img)
    # cv2.waitKey(0)
    # print(img)
    temp_file = tempfile.NamedTemporaryFile(delete=False , suffix='.jpg')
    # print(temp_file.name)
    cv2.imwrite(temp_file.name, gray_img)
    # # print(temp_filename)
    file_name = set_image_dpi(file_path)
    print(file_name)
    gray_img = remove_noise_smooth(file_name)
    config  = r'--oem 1 --psm 6 outoutbase digits'
    num_boxes = pytesseract.pytesseract.image_to_data(gray_img ,lang= lang, config=config)
    text = clean_text(pytesseract.pytesseract.image_to_string(gray_img, lang= lang))
    print(text)

    





# convert_gray_color('./image/id_6.jpg')
rectangle_detect('./image/id_4.jpg')
# set_image_dpi('./image/id_5.jpg')
# file_name = set_image_dpi('./image/id_6.jpg')
# remove_noise_smooth(file_name)

