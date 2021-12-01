import cv2
import matplotlib.pyplot as plt
import os
import tqdm 



def reverse_img(data_path):
    img_path_list = os.listdir(data_path)
    count = 0
    for img_path in img_path_list:
        count += 1
        img = cv2.imread(os.path.join(data_path, img_path))
        
        img = cv2.flip(img, 0)
        cv2.imwrite(os.path.join(data_path, img_path), img)
        print(img_path+" has been reversed, count "+str(count))


def filter_img(data_path):
    img_path_list = os.listdir(data_path)
    
    count = 0

    for img_path in img_path_list:
       
        img = cv2.imread(os.path.join(data_path, img_path)) # BGR
        mean_green = img[...,1].mean()
        if mean_green < 100.:
            count += 1
            # cv2.imshow('1', img)
            # cv2.waitKey(0)
            os.remove(os.path.join(data_path, img_path))
    
            print(img_path+" has been removed. count:"+str(count))


def remove_img():

    img_path = '543,541,537,516,515,509,506,472,471,466,464,460,453,451,448,447,446,438,433,430,427,421,420,380,383,333,350,286,169,101'
    img_path = img_path.split(',')
    img_path = [s+'.jpg' for s in img_path]

    root = '/shared/xiaolonw/yanjie/NUG-DLSS/data/fifa2real/train/B'
    for one_path in img_path:
        pt = os.path.join(root, one_path)
        os.remove(pt)
        print('remove %s success'%one_path)

if __name__=='__main__':
    # data_path ='data/fifa2real/train/A'
    # data_path ='data/fifa2real/test/A'
    # reverse_img(data_path)

    # data_path ='data/fifa2real/train/B'
    # data_path ='data/fifa2real/test/B'
    # filter_img(data_path)
    remove_img()
    