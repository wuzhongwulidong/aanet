import numpy as np
import skimage.io
from PIL import Image

# skimage.io.imsave(os.path.join(args.save_dir, left_image_path.split('/')
# [-1]), (disparity * 256).astype('uint16'))


import sys
# 图像的各种读取和保存方法参见：https://blog.csdn.net/huanghaocs/article/details/88411473
# https://www.it1352.com/753742.html 强烈推荐。

def read_img(filename):
    # Convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB'))
    return img


def main():
    img_npArray_1 = read_img("./myDemoData/000000_10.png")
    img_npArray_2 = read_img("./myDemoData/000000_11.png")

    img_npArray_1_1 = img_npArray_1 * 1 / 3
    img_npArray_1_2 = img_npArray_1 * 2 / 3

    # array to image
    image_output_1 = Image.fromarray(img_npArray_1_1)
    image_output_2 = Image.fromarray(img_npArray_1_2)
    # save image
    image_output_1.save("new_panda.png")
    image_output_2.save("new_panda.png")



    return


if __name__ == '__main__':
    main()