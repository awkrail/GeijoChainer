import numpy as np
import cv2
import os
import chainer
import chainer.functions as F
import chainer.links as L
import pandas as pd

faceCascade = cv2.CascadeClassifier("/Users/nishimurataichi/.pyenv/versions/anaconda3-4.1.0/pkgs/opencv3-3.1.0-py35_0/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")


def cut_out_face():
    img_paths = os.listdir('classmates')
    for img_path in img_paths:
        if '.DS_Store' in img_path:
            continue
        # import ipdb; ipdb.set_trace()
        img = cv2.imread('classmates/' + img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = faceCascade.detectMultiScale(gray, 1.1, 3)
        if len(face) > 0:
            for rect in face:
                x = rect[0]
                y = rect[1]
                width = rect[2]
                height = rect[3]
                dst = img[y:y + height, x:x + width]
                fixed_dst = cv2.resize(dst, (75, 75))
                new_path = 'classmates_face/' + img_path
                cv2.imwrite(new_path, fixed_dst)


def load_classmates():
    img_paths = os.listdir('classmates_face')
    for img_path in img_paths:
        if '.DS_Store' in img_path:
            continue
        img = cv2.imread('classmates_face/' + img_path)
        yield img_path, img


def img2numpy(fixed_dst):
    '''
    この関数,もっと高速に手順を減らすことができそう
    :return: imgからnumpyへ変換して返す
    '''
    img = np.asarray(fixed_dst, dtype=np.int8)
    r_img = []
    g_img = []
    b_img = []

    for i in range(75):
        for j in range(75):
            r_img.append(img[i][j][0])
            g_img.append(img[i][j][1])
            b_img.append(img[i][j][2])

    all_ary = r_img + g_img + b_img
    all_np = np.array(all_ary, dtype=np.float32).reshape(3, 5625)
    r, g, b = all_np[0], all_np[1], all_np[2]
    rImg = np.asarray(np.float32(r) / 255.0).reshape(75, 75)
    gImg = np.asarray(np.float32(g) / 255.0).reshape(75, 75)
    bImg = np.asarray(np.float32(b) / 255.0).reshape(75, 75)

    rgb = np.asarray([rImg, gImg, bImg]).reshape(1, 3, 75, 75)

    return rgb


class SimpleAlex(chainer.Chain):
    def __init__(self):
        super(SimpleAlex, self).__init__(
            conv1 = L.Convolution2D(None, 50, 6, stride=3),
            conv2 = L.Convolution2D(None, 75, 3, pad=1),
            fc1 = L.Linear(None, 200),
            fc2 = L.Linear(None, 11)
		)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=1)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=1)
        h = F.dropout(F.relu(self.fc1(h)))
        h = self.fc2(h)

        return h

professors = {
        0: 'ito',
        1: 'inamura',
        2: 'ushiama',
        3: 'kimu',
        4: 'takagi',
        5: 'ueoka',
        6: 'oshima',
        7: 'tomotani',
        8: 'tsuruno',
        9: 'fuyuno',
        10: 'fujihara',
        11: 'tomimatsu'
}


# cut_out_face()
if __name__ == '__main__':
    image_names = []
    professor_numbers = []

    for img_path, img in load_classmates():
        img = img2numpy(img)
        model = SimpleAlex()
        chainer.serializers.load_npz('geijo_result.npz', model)
        y = model(img)
        y = np.argmax(F.softmax(y).data)

        print(y)

        image_names.append(img_path)
        professor_numbers.append(professors[y])

    all_data = [image_names, professor_numbers]

    result = pd.DataFrame(all_data)
    result = result.T
    result.columns = ['学生名', '研究室名']
    result.to_csv('your_fate.csv', index=False)







