import cv2
import os
import numpy as np
import random
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
            img = cv2.imread('face_images/' + self.directory + '/' + fname + \
                    ext)
            flipped_img = cv2.flip(img, 1)
            # import ipdb; ipdb.set_trace()
            cv2.imwrite('af_face_images/' + self.directory + '/' + fname + '/' \
                    + ext, img)
            cv2.imwrite('af_face_images/' + self.directory + '/' + fname + '/' \
                    '_yAxis' + ext, flipped_img)
    
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
        
        i = 0
        for fname, ext in zip(self.fnames, self.exts):
            img = cv2.imread('face_images/' + self.directory + '/' + fname + \
                    ext)

            for matrix in matrixs:
                affine_img = cv2.warpAffine(img, matrix, size,
                    flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
                cv2.imwrite('af_face_images/' + self.directory + '/' + \
                    fname + '_afn%d' % i + ext, affine_img)
                flip_affine_img = cv2.warpAffine(cv2.flip(img, 1), matrix, size,
                    flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
                cv2.imwrite('af_face_images/' + self.directory + '/' + \
                    fname + '_yAxis_afn%d' % i + ext, flip_affine_img)

def main():
    directories = os.listdir('face_images')
    professors = []
    for directory in directories:
        if '.DS_Store' in directory:
            continue
        paths = [get_path(name) for name in os.listdir('face_images/' + directory + '/')
            if not '.DS_Store' in directory]
        exts = [get_ext(name) for name in os.listdir('face_images/' + directory + '/')
            if not '.DS_Store' in directory]
        professors.append(Augmentation(directory, paths, exts))

    for professor in professors:
        professor.flip()
        professor.affine()

def get_ext(path):
    _, ext = os.path.splitext(path)
    return ext

def get_path(path):
    name, _ = os.path.splitext(path)
    return name

if __name__ == '__main__':
    main()
