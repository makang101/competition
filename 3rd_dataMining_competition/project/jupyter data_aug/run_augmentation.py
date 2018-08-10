import os
import argparse
import random
import math
import numpy as np
from multiprocessing import Process
from multiprocessing import cpu_count
import cv2 as cv
import image_augmentation as ia


def parse_args():
    parser = argparse.ArgumentParser(
        description='A simple image data augment tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='directory containing images')
    parser.add_argument('output_dir', help='directory for augmented images')
    parser.add_argument('num', help='number of images to be augmented', type=int)
    parser.add_argument('--num_procs', help='Number of processes for paralleled augmentation', type=int, default=4)
    parser.add_argument('--p_mirror', help='ratio to mirror an image', type=float, default=0.5)
    parser.add_argument('--p_crop',
                        help='Ratio to randomly crop an image',
                        type=float, default=1.0)
    parser.add_argument('--crop_size', help='ratio of cropped image size to original image size,in area', type=float,
                        default=0.8)
    parser.add_argument('--crop_hw_vari', help='variation of h/w ratio', type=float, default=0.1)
    parser.add_argument('--p_rotate', help='ratio to randomly rotate an image', type=float, default=1.0)
    parser.add_argument('--p_rotate_crop', help='ratio to crop out the empty part in a rotated image', type=float,
                        default=1.0)
    parser.add_argument('--rotate_angle_vari', help='variation range of rotate angle', type=float, default=10.0)
    parser.add_argument('--p_hsv',
                        help='Ratio to randomly change gamma of an image',
                        type=float, default=1.0)
    parser.add_argument('--hue_vari',
                        help='Variation of hue',
                        type=int, default=10)
    parser.add_argument('--sat_vari',
                        help='Variation of saturation',
                        type=float, default=0.1)
    parser.add_argument('--val_vari',
                        help='Variation of value',
                        type=float, default=0.1)

    parser.add_argument('--p_gamma',
                        help='Ratio to randomly change gamma of an image',
                        type=float, default=1.0)
    parser.add_argument('--gamma_vari',
                        help='Variation of gamma',
                        type=float, default=2.0)

    args = parser.parse_args()
    args.input_dir = args.input_dir.rstrip('/')
    args.output_dir = args.output_dir.rstrip('/')
    return args

def generate_image_list(args,input_dir):
    filenames = os.listdir(input_dir)
    num_imgs = len(filenames)

    num_ave_aug = int(math.floor(args.num/num_imgs))
    rem = args.num - num_ave_aug*num_imgs
    lucky_seq = [True]*rem + [False]*(num_imgs-rem)
    random.shuffle(lucky_seq)

    img_list = [
        (os.sep.join([input_dir, filename]), num_ave_aug+1 if lucky else num_ave_aug)
        for filename, lucky in zip(filenames, lucky_seq)
    ]

    random.shuffle(img_list)  # in case the file size are not uniformly distributed

    length = float(num_imgs) / float(args.num_procs)
    indices = [int(round(i * length)) for i in range(args.num_procs + 1)]
    return [img_list[indices[i]:indices[i + 1]] for i in range(args.num_procs)]

def cv_imread(file_path):
    cv_img = cv.imdecode(np.fromfile(file_path, dtype = np.uint8), -1)
    return cv_img

def augment_images(filelist, args):
    for filepath, n in filelist:
        filepath = filepath.replace('\\','/')
        img = cv_imread(filepath)
        #print(filepath)
        #print(img)
        filename = filepath.split('/')[-1]
        dot_pos = filename.rfind('.')
        imgname = filename[:dot_pos]
        ext = filename[dot_pos:]
        #print(ext, imgname,filename)

        print('Augmenting {} ...'.format(filename))
        for i in range(n):
            img_varied = img.copy()
            varied_imgname = '{}_{:0>3d}_'.format(imgname, i)
            if random.random() < args.p_mirror:
                img_varied = cv.flip(img_varied, 1)
                varied_imgname += 'm'
            if random.random() < args.p_crop:
                img_varied = ia.random_crop(
                    img_varied,
                    args.crop_size,
                    args.crop_hw_vari)
                varied_imgname += 'c'
            if random.random() < args.p_rotate:
                img_varied = ia.random_rotate(
                    img_varied,
                    args.rotate_angle_vari,
                    args.p_rotate_crop)
                varied_imgname += 'r'
            if random.random() < args.p_hsv:
                img_varied = ia.random_hsv_transform(
                    img_varied,
                    args.hue_vari,
                    args.sat_vari,
                    args.val_vari)
                varied_imgname += 'h'
            if random.random() < args.p_gamma:
                img_varied = ia.random_gamma_transform(
                    img_varied,
                    args.gamma_vari)
                varied_imgname += 'g'
            output_filepath = os.sep.join([args.output_dir,'{}{}'.format(varied_imgname, ext)]).replace('\\','/')
            #cv.imwrite(output_filepath, img_varied)

            cv.imencode('.png', img_varied)[1].tofile(output_filepath)

def gothrough(filepath,filepathsets):
    filelist = os.listdir(filepath)
    for i in range(len(filelist)):
        filename = filelist[i]
        #print(str(i) + ':' + filename)
        cur_filepath = (filepath + '/' + filename)
        #print(filepath)
        if os.path.isdir(cur_filepath):
            gothrough(cur_filepath, filepathsets)
        else:
            filepathsets.append(filepath)
            #print(filepath)
            break
    return filepathsets

def main():
    args = parse_args()
    params_str = str(args)[10:-1]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print('Starting image data augmentation for {}\n'
          'with\n{}\n'.format(args.input_dir, params_str))
    filepathsets = []
    filepathsets = gothrough(args.input_dir, filepathsets)
    i = 0
    for input_dir in filepathsets:
        i = i + 1
        sublists = generate_image_list(args,input_dir)
        processes = [Process(target=augment_images, args=(x, args, )) for x in sublists]

        for p in processes:
            p.start()

        for p in processes:
            p.join()
        if i % 50:
            print('第{}个文件夹已经处理完了！'.format(i))

if __name__ == '__main__':
    main()