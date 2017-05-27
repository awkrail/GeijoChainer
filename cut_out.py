import os
import cv2
from PIL import Image
import json
import numpy as np
import keras.preprocessing.image as kerasimage
import random
import shutil

img_path = os.path.normpath(os.path.join(os.path.abspath(__name__), 'rakuten_hairImages'))
json_path = os.path.normpath(os.path.join(os.path.abspath(__name__), 'rhair.json'))
faceCascade = cv2.CascadeClassifier("/Users/nishimurataichi/.pyenv/versions/anaconda3-4.1.0/pkgs/opencv3-3.1.0-py35_0/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")


def change_name():
	images = os.listdir('rakuten_hairImages')
	for image in images:
		if image[0] == '.':
			continue
		img = Image.open('rakuten_hairImages/' + image)
		new_name = 'r' + image
		# 画像の名前を変更する
		img.save('./new_name_images/' + new_name)

def change_json_key():
	dict_ary = []
	with open('rhair.json', 'r') as f:
		json_ary = json.load(f)
		#print(json_ary)
		for json_data in json_ary["hair"]:
			url = json_data['url']
			file_id = json_data['fileId']
			new_dict = {}
			new_dict['url'] = url
			new_dict['file_id'] = 'r' + file_id
			dict_ary.append(new_dict)

	inclusive_dict = {}
	inclusive_dict['hair'] = dict_ary

	with open('new_rhair.json', 'w') as f:
		json.dump(inclusive_dict, f, ensure_ascii=False, sort_keys=True, separators=(',', ': '))

def recognized_hair():
	recognized_girls = []
	# print(os.listdir('rakuten/new_name_images'))
	for image in os.listdir('rakuten_hairImages'):
		if image.find('.DS_Store') > -1:
			 continue
		img = cv2.imread('rakuten_hairImages/' + image, cv2.IMREAD_COLOR)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		face = faceCascade.detectMultiScale(gray, 1.1, 3)
		if len(face) > 0:
			recognized_girls.append(image)

	print('recognized_girls_counts:' + str(len(recognized_girls)))

	for girl in recognized_girls:
		path = 'rakuten_hairImages/' + girl
		img = Image.open(path)
		new_name = 'r' + girl
		img.save('recognized_girls/' + new_name)

def check_counts():
	directories = os.listdir('2_8_images')
	for directory in directories:
		if directory.find('.DS_Store') > -1:
			continue
		img_ary = os.listdir('2_8_images/' + directory)
		print(directory + 'の枚数:' + str(len(img_ary)))

def cut_out_face():
	directories = os.listdir('2_8_images')
	for directory in directories:
		if directory.find('.DS_Store') > -1:
			continue

		img_ary = os.listdir('2_8_images/' + directory)
		for image in img_ary:
			if image.find('.DS_Store') > -1:
				continue

			img = cv2.imread('2_8_images/' + directory + '/' + image, cv2.IMREAD_COLOR)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			face = faceCascade.detectMultiScale(gray, 1.1, 3)
			if len(face) > 0:
				for rect in face:
					x = rect[0]
					y = rect[1]
					width = rect[2]
					height = rect[3]
					dst = img[y:y+height, x:x+width]
					fixed_dst = cv2.resize(dst, (75,75))

					if not os.path.exists('hair2/' + directory):
						os.mkdir('hair2/' + directory)

					np_path = 'hair2/' + directory + '/' + image
					cv2.imwrite(np_path, fixed_dst)
					# np_image = np.

def tsubasa():
	tsubasa_img = cv2.imread('tsubasa.jpeg', cv2.IMREAD_COLOR)
	gray = cv2.cvtColor(tsubasa_img, cv2.COLOR_BGR2GRAY)
	face = faceCascade.detectMultiScale(gray, 1.1, 3)
	if len(face) > 0:
		for rect in face:
			x = rect[0]
			y = rect[1]
			width = rect[2]
			height = rect[3]
			dst = tsubasa_img[y:y+height, x:x+width]
			fixed_dst = cv2.resize(dst, (75,75))

	np_path = 'tsubasa_75.jpg'
	cv2.imwrite(np_path, fixed_dst)


# OpenCVでデータ拡張 & pickle化するコードを書く
def img2pickle_openCV():
	directories = os.listdir('hair2')
	for directory in directories:
		if directory.find('.DS_Store') > -1:
			continue

		for img in os.listdir('hair2/' + directory):
			if img.find('.DS_Store') > -1:
				continue

			#注目の画像
			cv_img = cv2.imread('hair2/' + directory + '/' + img, cv2.IMREAD_COLOR)
			# y軸反転した画像
			if img.find('jpg') > -1:
				new_name = img.replace('.jpg', '')
			elif img.find('png') > -1:
				new_name = img.replace('.png', '')
			elif img.find('jpeg') > -1:
				new_name = img.replace('.jpeg', '')
			else:
				new_name = img

			yAxis = cv2.flip(cv_img, 1)
			path = 'more_ag_pickle/' + directory + '/' + new_name
			#元のデータ,左右反転データをいれておく
			cv2.imwrite(path + '.jpg',cv_img)
			cv2.imwrite(path + '_yAxis.jpg',yAxis)

			#少し傾けたデータを入れる(10~20度ほど,ランダムに)
			size = (75,75)

			#回転させたい角度
			rad1 = np.pi / random.randint(-30,-10)
			rad2 = np.pi / random.randint(10,30)
			rad3 = np.pi / random.randint(-30,-10)
			rad4 = np.pi / random.randint(10,30)

			move_x = 0
			move_y = 0

			matrix1 = [
						[np.cos(rad1),	-1*np.sin(rad1), move_x],
						[np.sin(rad1),	np.cos(rad1), move_y]
					]
			affine_matrix1 = np.float32(matrix1)
			img_afn1 = cv2.warpAffine(cv_img, affine_matrix1, size, flags=cv2.INTER_LINEAR)
			cv2.imwrite(path + '_affine1.jpg',img_afn1)

			matrix2 = [
						[np.cos(rad2),	-1*np.sin(rad2), 0],
						[np.sin(rad2),	np.cos(rad2), 0]
					]

			affine_matrix2 = np.float32(matrix2)
			img_afn2 = cv2.warpAffine(cv_img, affine_matrix2, size, flags=cv2.INTER_LINEAR)
			cv2.imwrite(path + '_affine2.jpg',img_afn2)

			matrix3 = [
						[np.cos(rad3), -1*np.sin(rad3), 0],
						[np.sin(rad3), np.cos(rad3), 0]
				]
			affine_matrix3 = np.float32(matrix3)
			img_afn3 = cv2.warpAffine(cv_img, affine_matrix3, size, flags=cv2.INTER_LINEAR)
			cv2.imwrite(path + '_affine3.jpg', img_afn3)

			matrix4 = [
						[np.cos(rad4), -1*np.sin(rad4), 0],
						[np.sin(rad4), np.cos(rad4), 0]
			]
			affine_matrix4 = np.float32(matrix4)
			img_afn4 = cv2.warpAffine(cv_img, affine_matrix4, size, flags=cv2.INTER_LINEAR)
			cv2.imwrite(path + '_affine4.jpg', img_afn4)

def img2pickle_withZCA():
	directories = os.listdir('dec_noise_75_pickle')
	for directory in directories:
		if directory.find('.DS_Store') > -1:
			continue
		for image in os.listdir('dec_noise_75_pickle/' + directory):
			if image.find('.DS_Store') > -1:
				continue
			path = 'dec_noise_75_pickle/' + directory + '/' + image
			np_img = np.load(path)
			print(np_img.shape[1])

			sigma = np.dot(np_img, np_img.T)/np_img[1]
			U,S,V = np.linalg.svd(sigma)
			epsilon = 0.1
			ZCA_Matrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)
			data = np.dot(ZCA_Matrix, np_img)

			if image.find('npy') > -1:
				new_name = image.replace('.npy', '')

			np_path = 'zca_hair_pickle/' + directory + '/' + new_name
			np.save(np_path, data)


def img2pickle_af_augument():
	directories = os.listdir('hair2')
	for directory in directories:
		if directory.find('.DS_Store') > -1:
			continue
		for image in os.listdir('more_ag_pickle/' + directory):
			if image.find('.DS_Store') > -1:
				continue

			np_img = np.asarray(Image.open('more_ag_pickle/' + directory + '/' +image).convert('RGB'), dtype=np.uint8)
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

			if image.find('jpg') > -1:
				new_name = image.replace('.jpg', '')
			elif image.find('png') > -1:
				new_name = image.replace('.png', '')
			elif image.find('jpeg') > -1:
				new_name = image.replace('.jpeg', '')
			else:
				new_name = image

			np_path = 'hair_pickle/' + directory + '/' + new_name
			np.save(np_path, all_np)


def img2pickle():
	directories = os.listdir('hair_pickle')
	for directory in directories:
		if directory.find('.DS_Store') > -1:
			continue

		img_ary = os.listdir('hair_pickle/' + directory)
		for image in img_ary:
			if image.find('.DS_Store') > -1:
				continue

			# 画像データ
			img = Image.open('hair_pickle/' + directory + '/' + image)
			# 元データ
			np_img = np.asarray(Image.open('hair_pickle/' + directory + '/' +image).convert('RGB'), dtype=np.uint8)
			aug_ary = []

			np_img = np.asarray(np.float32(np_img) / 255.0)
			aug_ary.append(np_img)

			for i in range(5):
				x = np.expand_dims(np_img, axis=0)
				temp_dir = "temp"
				os.mkdir(temp_dir)
				p1 = random.random()
				p2 = random.random()
				p3 = random.random()
				p4 = random.random()
				p5 = random.random()
				augdata = kerasimage.ImageDataGenerator(
						horizontal_flip = True,
						vertical_flip = True
					)
				aug_ks = augdata.flow(x, batch_size=1, save_to_dir=temp_dir, save_prefix='aug_img', save_format='jpg')
				aug_ks.next()
				aug_img = Image.open('temp/' + os.listdir('temp')[0])
				aug_np = np.asarray(aug_img, dtype=np.float32)
				aug_np = np.asarray(np.float32(aug_np) / 255.0)
				# ちょっと議論の余地あり
				aug_ary.append(aug_np)

				#ディレクトリ削除
				shutil.rmtree(temp_dir)

			# 拡張したデータの分,データを更に細かいディレクトリにして管理する
			for i,np_data in enumerate(aug_ary):
				r_img = []
				g_img = []
				b_img = []
				for i in range(32):
					for j in range(32):
						r_img.append(np_data[i][j][0])
						g_img.append(np_data[i][j][1])
						b_img.append(np_data[i][j][2])

				all_ary = r_img + g_img + b_img
				all_np = np.asarray(all_ary, dtype=np.float32)

				if image.find('jpg') > -1:
					new_name = image.replace('.jpg', '')
				elif image.find('png') > -1:
					new_name = image.replace('.png', '')
				elif image.find('jpeg') > -1:
					new_name = image.replace('.jpeg', '')
				else:
					new_name = image

				np_path = 'new_pickle/' + directory + '/' + new_name + '_' + str(i)
				np.save(np_path, all_np)

def decrease_pickle():
	directories = os.listdir('hair_pickle')
	for directory in directories:
		if directory.find('.DS_Store') > -1:
			continue
		for image in os.listdir('hair_pickle/' + directory):
			if image.find('.DS_Store') > -1:
				continue

			if image.find('_affine') > -1 or image.find('_yAxis') > -1:
				continue

			np_img = np.load('hair_pickle/' + directory + '/' + image)
			new_name = image.replace('.npy', '')
			np_path = 'result_pickle/' + new_name

			np.save(np_path, np_img)


def decrease_pickle_with_label():
	directories = os.listdir('hair_pickle')
	for directory in directories:
		if directory.find('.DS_Store') > -1:
			continue

		for image in os.listdir('hair_pickle/' + directory):
			if image.find('.DS_Store') > -1:
				continue

			if image.find('_affine') > -1 or image.find('_yAxis') > -1:
				continue

			np_img = np.load('hair_pickle/' + directory + '/' + image)
			new_name = image.replace('.npy', '')
			np_path = 'decrease_pickle/' + directory + '/' + new_name

			np.save(np_path, np_img)


def change_json_key():
	with open('hotpepper.json', 'r') as f:
		dicts = json.load(f)

	for dict in dicts:
		if 2346 <= dict['id'] <= 2859:
			dict['id'] -= 704
		elif 4669 <= dict['id'] <= 5213:
			dict['id'] -= 4669
		elif 3101 <= dict['id'] <= 3596:
			dict['id'] -= 1966
		elif 3846 <= dict['id'] <= 4426:
			dict['id'] -= 3294
		elif 1637 <= dict['id'] <= 2096:
			dict['id'] += 524

	with open('hotpepper2.json', 'w') as f:
		json.dump(dicts, f)



#change_name()
#change_json_key()
#recognized_hair()
#check_counts()
#cut_out_face()
#img2pickle_openCV()
#img2pickle_af_augument()
#img2pickle_withZCA()
#img2pickle()
#decrease_pickle_with_label()
change_json_key()

