import cv2
import os.path
import numpy as np
import random
"""
    data拡張:
    (1) 元データ
    (2) 元データ*5に拡張したもの
    (3) 反転データ
    (4) 反転データ*5に拡張したもの
"""

def main():
    pic_iter = zip_line_and_pic()

    for pic, line, pokepix in pic_iter:
        zip_write_org(pic, line, pokemon_name=pokepix)
        flip_color_pic, flip_line_pic = zip_flip(pic, line, pokemon_name=pokepix)

        # オリジナル画像でアフィン変換
        zip_affine(pic, line, pokemon_name=pokepix)

        # 左右反転後の画像でアフィン変換
        zip_affine(flip_color_pic, flip_line_pic, pokemon_name=pokepix, flag=False)


def zip_line_and_pic():
    """
    :return: pic_img,line_img: cv2で読み込まれたデータ. pokepix:ポケモンの名前(画像名)
    """
    for pokepix, line in zip(os.listdir('poke_512/af_pokepix512/'), os.listdir('poke_512/line')):
        if pokepix.find('.DS_Store') > -1 and line.find('.DS_Store') > -1:
            continue
        pic_img = cv2.imread('poke_512/af_pokepix512/' + pokepix)
        line_img = cv2.imread('poke_512/line/' + pokepix)

        yield pic_img, line_img, pokepix


def zip_write_org(color_pic, line_pic, pokemon_name):
    cv2.imwrite('test_space/label/' + pokemon_name, color_pic)
    cv2.imwrite('test_space/train/' + pokemon_name, line_pic)


def zip_flip(color_pic, line_pic, pokemon_name):

    color_pic_flip = cv2.flip(color_pic, 1)
    line_pic_flip = cv2.flip(line_pic, 1)

    cv2.imwrite('test_space/label/' + remove_png(pokemon_name) + '_yAxis.png', color_pic_flip)
    cv2.imwrite('test_space/train/' + remove_png(pokemon_name) + '_yAxis.png', line_pic_flip)

    return color_pic_flip, line_pic_flip


def zip_affine(color_pic, line_pic, pokemon_name, flag=True):

    size = (512, 512)

    # 回転角度と行列の配列定義
    rad1 = np.pi / random.randint(-30, -10)
    rad2 = np.pi / random.randint(10, 30)
    rad3 = np.pi / random.randint(-30, -10)
    rad4 = np.pi / random.randint(10, 30)
    rad5 = np.pi / random.randint(-30, -10)
    rads = [rad1, rad2, rad3, rad4, rad5]

    matrixs = [np.float32([
        [np.cos(rad), -1 * np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0]
               ]) for rad in rads]

    for i, matrix in enumerate(matrixs):
        line_img_afn = cv2.warpAffine(line_pic, matrix, size, flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        color_img_afn = cv2.warpAffine(color_pic, matrix, size, flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        if flag:
            cv2.imwrite('test_space/label/' + remove_png(pokemon_name) + '_afn%d' % i + '.png', color_img_afn)
            cv2.imwrite('test_space/train/' + remove_png(pokemon_name) + '_afn%d' % i + '.png', line_img_afn)
        else:
            cv2.imwrite('test_space/label/' + remove_png(pokemon_name) + '_yAxis_afn%d' % i + '.png', color_img_afn)
            cv2.imwrite('test_space/train/' + remove_png(pokemon_name) + '_yAxis_afn%d' % i + '.png', line_img_afn)


def remove_png(name):
    _, ext = os.path.splitext(name)
    name = name.replace(ext, '')

    return name

if __name__ == '__main__':
    main()
