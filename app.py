from flask import Flask, redirect, url_for, render_template, request
from werkzeug.utils import secure_filename
import os
from uuid import uuid4
import os
import cv2
import math
import model
import imutils
import scipy.misc
import numpy as np
from PIL import Image
from subprocess import call
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
sess = tf.InteractiveSession()
saver = tf.train.Saver()

saver.restore(sess, "model.ckpt")

# Converting video to images
path = './deploy/video2image/'
path1 = './deploy/steering_image/'
path2 = './deploy/merged_image/'


app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "./static/upload"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/uploader", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        unique_key = str(uuid4())
        filename = unique_key + filename
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_path = "upload/" + filename
        # Converting video to images
        vidcap = cv2.VideoCapture('min_video.mp4')

        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
            hasFrames, image = vidcap.read()
            if hasFrames:
                # save frame as JPG file
                cv2.imwrite(path+str(count)+".jpg", image)
            return hasFrames

        sec = 0
        frameRate = 1  # //it will capture image in each 0.5 second
        count = 1
        success = getFrame(sec)
        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec)

        lis = os.listdir(path)

        img1 = cv2.imread('steering_wheel_image.jpg', 0)
        rows, cols = img1.shape

        smoothed_angle = 0
        i = 1
        count = 1
        # Predict and transforming operation(rotation)
        for i in lis:
            full_image = scipy.misc.imread(path + i, mode="RGB")
            image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
            degrees = model.y.eval(session=sess, feed_dict={model.x: [image], model.keep_prob: 1.0})[
                0][0] * 180.0 / scipy.pi
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (
                degrees - smoothed_angle) / abs(degrees - smoothed_angle)
            M = cv2.getRotationMatrix2D((cols/2, rows/2), -smoothed_angle, 1)
            img2 = cv2.warpAffine(img1, M, (cols, rows))
            cv2.imwrite(path1 + str(count)+".jpg", img2)
            count += 1

        # ### resizing all image and concatening

        counter = 1
        zero = 0
        for j in range(1, (len(lis)+1)):
            img3 = cv2.imread(path + str(j)+".jpg")
            img4 = cv2.imread(path1 + str(j)+".jpg")
            scale_percent = 80  # percent of original size
            width = int(img4.shape[1] * scale_percent / 100)
            height = int(img4.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            img4 = cv2.resize(img4, dim, interpolation=cv2.INTER_AREA)
            rows, cols, channels = img4.shape
            roi = img3[0:rows, 0:cols]
            # Now create a mask of logo and create its inverse mask also
            img2gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            # Now black-out the area of logo in ROI
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            # Take only region of logo from logo image.
            img2_fg = cv2.bitwise_and(img4, img4, mask=mask)
            # Put logo in ROI and modify the main image
            dst = cv2.add(img1_bg, img2_fg)
            img3[0:rows, 0:cols] = dst
            # cv2.imshow("self_ride",img3)
            cv2.imwrite(path2 + str(counter) + ".jpg", img3)
            counter += 1
            zero += 1
        cv2.destroyAllWindows()

        # Creating video file from images
        image_folder = path2
        video_name = f'./static/result/{unique_key}video.webm'
        result_video = f'result/{unique_key}video.webm'
        # video_name = 'result/' + filename
        images = [img for img in os.listdir(
            image_folder) if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, 0x30385056, 1, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

        dir1 = './deploy/merged_image'
        for i in os.listdir(dir1):
            os.remove(os.path.join(dir1, i))

        dir2 = './deploy/steering_image'
        for i in os.listdir(dir2):
            os.remove(os.path.join(dir2, i))

        dir3 = './deploy/video2image'
        for i in os.listdir(dir3):
            os.remove(os.path.join(dir3, i))

        return render_template("result.html", value=result_video)


if __name__ == '__main__':
    app.run(debug=True)
