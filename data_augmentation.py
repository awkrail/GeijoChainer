import cv2
import os
import numpy as np
import random
from PIL import Image

"""
    data拡張:
    (1)元データ
    (2)元データ*5に拡張したもの
    (3)反転データ
    (4)反転データ*5に拡張したもの
    TODO: クラスをつくってディレクトリごとに管理すればokでは
"""


class Augmentation(object):

    def __init__(self, directory, fnames, exts):
        self.directory = directory
        self.fnames = fnames
        self.exts = exts

    # 保存してflipしたものも保存する
    def flip(self):
        for fname, ext in zip(self.fnames, self.exts):
            img = cv2.imread('face_images/' + self.directory + '/' + fname + ext)
            flipped_img = cv2.flip(img, 1)
            # import ipdb; ipdb.set_trace()
            cv2.imwrite('af_face_images/' + self.directory + '/' + fname + ext, img)
            cv2.imwrite('af_face_images/' + self.directory + '/' + fname + '_yAxis' + ext, flipped_img)

    def affine(self):
        size = (75, 75)
        
        rad1 = np.pi / random.randint(-30, -10)
        rad2 = np.pi / random.randint(10, 30)
        rad3 = np.pi / random.randint(-30, -10)
        rad4 = np.pi / random.randint(10, 30)
        rad5 = np.pi / random.randint(-30, -10)
        rads = [rad1, rad2, rad3, rad4, rad5]

        matrixs = [np.float32([
            [np.cos(rad), -1 * np.sin(rad), 0],
            [np.sin(rad), np.cos(rad), 0]]) for rad in rads]

        for fname, ext in zip(self.fnames, self.exts):
            img = cv2.imread('face_images/' + self.directory + '/' + fname + ext)
            # import ipdb; ipdb.set_trace()

            for i, matrix in enumerate(matrixs):
                affine_img = cv2.warpAffine(img, matrix, size, flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
                cv2.imwrite('af_face_images/' + self.directory + '/' + fname + '_afn%d' % i + ext, affine_img)
                # for reading pickle data
                flip_affine_img = cv2.warpAffine(cv2.flip(img, 1), matrix, size, flags=cv2.INTER_LINEAR,
                                                 borderValue=(255, 255, 255))
                cv2.imwrite('af_face_images/' + self.directory + '/' + fname + '_yAxis_afn%d' % i + ext,
                            flip_affine_img)

    def make_pickle(self):
        img_paths = os.listdir('af_face_images/' + self.directory)
        for img_path in img_paths:
            if '.DS_Store' in img_path:
                continue
            img = Image.open('af_face_images/' + self.directory + '/' + img_path).convert('RGB')
            fname = get_path(img_path) # 拡張子を外す
            np_img = np.array(img, dtype=np.uint8)
            r_img = []
            g_img = []
            b_img = []

            for i in range(75):
                for j in range(75):
                    r_img.append(np_img[i][j][0])
                    g_img.append(np_img[i][j][1])
                    b_img.append(np_img[i][j][2])

            all_ary = r_img + g_img + b_img
            all_np = np.array(all_ary, dtype=np.float32)

            np_path = 'af_face_pickle/' + self.directory + '/' + fname

            if not os.path.exists('af_face_pickle/' + self.directory):
                os.mkdir('af_face_pickle/' + self.directory)

            np.save(np_path, all_np)


def main():
    directories = os.listdir('face_images')
    professors = []
    for directory in directories:
        if '.DS_Store' in directory:
            continue
        paths = [get_path(name) for name in os.listdir('face_images/' + directory + '/')
            if not '.DS_Store' in directory and not '.DS_Store' in name]
        exts = [get_ext(name) for name in os.listdir('face_images/' + directory + '/')
            if not '.DS_Store' in directory and not '.DS_Store' in name]
        professors.append(Augmentation(directory, paths, exts))

    for professor in professors:
        professor.flip()
        professor.affine()
        professor.make_pickle()


def get_ext(path):
    _, ext = os.path.splitext(path)
    return ext


def get_path(path):
    name, _ = os.path.splitext(path)
    return name

if __name__ == '__main__':
    main()
