import os
import cv2
faceCascade = cv2.CascadeClassifier("/Users/nishimurataichi/.pyenv/versions/anaconda3-4.1.0/pkgs/opencv3-3.1.0-py35_0/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")

def cut_out_face():
    directories = os.listdir('images')
    for directory in directories:
        print(directory)
        if 'DS_Store' in directory:
            continue
        img_paths = os.listdir('images/' + directory)
        for img_path in img_paths:
            if 'DS_Store' in img_path:
                continue
            img = cv2.imread('images/' + directory + '/' + img_path, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # import ipdb; ipdb.set_trace()
            face = faceCascade.detectMultiScale(gray, 1.1, 3)
            if len(face) > 0:
                for rect in face:
                    x = rect[0]
                    y = rect[1]
                    width = rect[2]
                    height = rect[3]
                    dst = img[y:y+height, x:x+width]
                    fixed_dst = cv2.resize(dst, (75, 75))
                    
                    if not os.path.exists('face_images/' + directory):
                        os.mkdir('face_images/' + directory)
                    new_path = 'face_images/' + directory + '/' + img_path
                    cv2.imwrite(new_path, fixed_dst)


if __name__ == '__main__':
    cut_out_face()
