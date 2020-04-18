import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from PIL import Image
from skimage import transform as tf
try:
    from .rigid_registration import write_params
except:
    from rigid_registration import write_params
from math import pi

def main():
    parser = argparse.ArgumentParser(description='Perform manual rigid registration.')
    parser.add_argument('-f','--fixed', help="Fixed image",metavar='<filename>')
    parser.add_argument('-m','--moving', help="Moving image",metavar='<filename>')
    parser.add_argument('-p','--params', help="Output parameter file",metavar='<filename>')
    parser.add_argument('-c','--cmp', help="Output comparison image filename",metavar='<filename>')
    parser.add_argument('--scale', help="Scaling factor",metavar='<n>',type=int,default=5)

    args = parser.parse_args()
    scale = args.scale
    fixed_original = np.array(Image.open(args.fixed))[:,:,0]
    fixed = np.round(tf.pyramid_reduce(fixed_original, scale)*255).astype(np.uint8)
    moving_original = np.array(Image.open(args.moving))[:,:,0]
    moving = np.round(tf.pyramid_reduce(moving_original, scale)*255).astype(np.uint8)
    moved = np.copy(moving)
    fixed_alpha = .5
    moving_alpha = .5
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)

    axcolor = 'lightgoldenrodyellow'
    axangle = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    axtx = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    axty = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    sangle = Slider(axangle, 'Angle', -pi, pi, valinit=0, valstep=pi/360)
    stx = Slider(axtx, 'X', -moving.shape[1], moving.shape[1], valinit=0, valstep=1)
    sty = Slider(axty, 'Y', -moving.shape[0], moving.shape[0], valinit=0, valstep=1)

    center = np.array(moving.shape[:2][::-1])/2-.5
    def imshow():
        ax.clear()
        ax.imshow(fixed,alpha=fixed_alpha,cmap='Reds_r',clim=(0,255))
        ax.imshow(moved,alpha=moving_alpha,cmap='Greens_r',clim=(0,255))
        fig.canvas.draw()

    def update(val):
        tx = stx.val
        ty = sty.val
        angle = sangle.val
        t = tf.EuclideanTransform(translation=-center) + tf.EuclideanTransform(rotation=angle,translation=(tx,ty)) + tf.EuclideanTransform(translation=center)
        nonlocal moved
        moved = np.round(tf.warp(moving, t.inverse, preserve_range=True)).astype(np.uint8)
        imshow()

    sangle.on_changed(update)
    stx.on_changed(update)
    sty.on_changed(update)

    okax = plt.axes([0.8, 0.05, 0.1, 0.04])
    ok_button = Button(okax, 'Ok', color=axcolor, hovercolor='0.975')
    cancelax = plt.axes([0.65, 0.05, 0.1, 0.04])
    cancel_button = Button(cancelax, 'Cancel', color=axcolor, hovercolor='0.975')


    def cancel(event):
        plt.close()

    def ok(event):
        plt.close()
        tx = stx.val
        ty = sty.val
        angle = sangle.val
        params = [-angle, ty*scale, -tx*scale]
        if args.params:
            write_params(args.params,params)
        else:
            print(params)

        if args.cmp:
            center = np.array(moving_original.shape[:2][::-1])/2-.5
            t = tf.EuclideanTransform(translation=-center) + tf.EuclideanTransform(rotation=angle,translation=(tx*scale,ty*scale)) + tf.EuclideanTransform(translation=center)
            moved = np.round(tf.warp(moving_original, t.inverse, preserve_range=True)).astype(np.uint8)
            Image.fromarray(np.stack((fixed_original, moved, np.zeros_like(fixed_original)), axis=-1)).save(args.cmp)
    ok_button.on_clicked(ok)
    cancel_button.on_clicked(cancel)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('fixed', 'moving', 'both'), active=2)

    def colorfunc(label):
        nonlocal fixed_alpha
        nonlocal moving_alpha
        if label=='fixed':
            fixed_alpha = 1
            moving_alpha = 0
        elif label=='moving':
            fixed_alpha = 0
            moving_alpha = 1
        else:
            fixed_alpha = .5
            moving_alpha = .5
        imshow()
    radio.on_clicked(colorfunc)

    update(0)
    imshow()
    plt.show()
    return 0

if __name__ == "__main__":
    sys.exit(main())
