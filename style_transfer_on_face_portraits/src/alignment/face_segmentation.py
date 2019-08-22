import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread
import surgery
import caffe
import cv2
import os.path
import click


# Face segmentation using the code from the "Deep face segmentation in extremely
# hard conditions" (https://github.com/YuvalNirkin/face_segmentation) related to
# the paper "On Face Segmentation, Face Swapping, and Face Perception" by Nirkin
# et al.
def segmentation(img_path):
    img = Image.open(img_path)

    # resize to fit the assumptions of the model
    size_ = 300
    img = img.resize((size_, size_))
    in_ = np.array(img, dtype=np.float32)

    result = imread(img_path)
    result = cv2.resize(result, dsize=(size_,size_))

    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # load a light version of the segmentation network
    net = caffe.Net(
        'data/face_seg_fcn8s_300_deploy.prototxt',
        'data/face_seg_fcn8s_300.caffemodel', caffe.TEST)

    # run the network, make a prediction
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    net.forward()
    prediction = net.blobs['score'].data[0].argmax(axis=0)

    output = np.zeros((size_, size_, 3))
    output[:,:,0] = output[:,:,1] = output[:,:,2] = prediction

    result[output == 0] = 0
    result = np.uint8(np.clip(result, 0, 255))

    result = Image.fromarray(result)
    result.putalpha(Image.fromarray(np.uint8(255 * prediction)).convert("L"))
    result.save(os.path.splitext(img_path)[0]+"_segmentation.png")

@click.command()
@click.option('--img')
def main(img):
    segmentation(img)

if __name__ == "__main__":
    main()
