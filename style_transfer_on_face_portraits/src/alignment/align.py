import facemorpher
import facemorpher.warper as warper
import facemorpher.locator as locator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import scipy.spatial as spatial
import click
import os.path

from helpers.image import Image


# example of execution:
# python3 align.py --content images/content.png --style images/style.png

# Detect faces present in both a style and content image, then warp the face
# from the style image to match the content and apply it on top of the content
# image.
#
# parameters:
#   content: path to the content image
#   style: path to the style image
#
# If the provided file names become, without extensions "content" and "style"
# for example, four images are generated:
#   style_aligned.png: the warped face from the style image
#   content_aligned_img: the modified content image (with face from the style
#                        image)
#   style_delaunay.png: a delaunay triangulation of the facial landmarks
#                       extracted from the style image
#   content_aligned_delaunay.png: the triangulation mentioned above, applied
#                                 to the aligned image.
def alignment(content, style):

    content_img = np.array(Image.from_file(content).data)
    style_img  = np.array(Image.from_file(style).data)
    content_name = os.path.splitext(content)[0]
    style_name = os.path.splitext(style)[0]


    # extraction of facial landmarks
    content_pts = locator.face_points(content_img, add_boundary_points=True)
    style_pts = locator.face_points(style_img, add_boundary_points=True)

    # translation in terms of coordinates
    content_coords = warper.grid_coordinates(content_pts)
    style_coords = warper.grid_coordinates(style_pts)

    # warp the face from the style image, given the extracted landmarks
    style_aligned_img = warper.warp_image(style_img, style_pts, content_pts, \
        content_img.shape)
    PIL.Image.fromarray(style_aligned_img).save(style_name+"_aligned.png")

    # apply the warped face from the style image on the content image
    mask = np.ma.masked_greater(style_aligned_img, 0)
    content_img[(mask!=0)] = 0
    content_aligned_img = style_aligned_img + content_img
    PIL.Image.fromarray(content_aligned_img).save(content_name+"_aligned.png")

    # obtain a delaunay triangulation of the style image's facial landmarks
    delaunay_style = spatial.Delaunay(style_pts)

    # visualize the delaunay triangulation of the style image's facial landmarks
    plt.triplot(style_pts[:,0], style_pts[:,1], delaunay_style.simplices.copy())
    plt.imshow(style_img)
    plt.plot(style_pts[:,0], style_pts[:,1], 'o')
    plt.savefig(style_name+"_delaunay.png")

    # apply the same triangulation to the new image, warped to fit the content
    # and visualize it
    plt.figure()
    plt.triplot(content_pts[:,0], content_pts[:,1], delaunay_style.simplices.copy())
    plt.imshow(content_aligned_img)
    plt.plot(content_pts[:,0], content_pts[:,1], 'o')
    plt.savefig(content_name+"_aligned_delaunay.png")

@click.command()
@click.option('--content')
@click.option('--style')
def main(content, style):
    alignment(content, style)

if __name__ == "__main__":
    main()
