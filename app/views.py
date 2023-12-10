import base64
import datetime
import re

import cv2
import numpy as np
import pickle5 as pickle
import tensorflow as tf
import tensorflow_hub as hub
from django.http import HttpResponse
from django.shortcuts import render, redirect
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pandas import read_csv

import app.admin


def about(request):
    if 'login' in request.session and request.session['login']:
        return render(request, 'app/about.html')
    else:
        return redirect(login)


def model(request):
    model_path = "D:/Models/tf2-preview_inception_v3_feature_vector_4"
    model = tf.keras.Sequential([
        hub.KerasLayer(hub.load(model_path), output_shape=[2048], trainable=False),
        tf.keras.layers.Dense(5, activation="sigmoid")
    ])
    model.build([None, 224, 224, 3])
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    model.load_weights('media/model/model.h5')
    with open('media/model.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    with open('media/model.txt', 'r') as fh:
        report = fh.read()
    if 'login' in request.session and request.session['login']:
        return render(request, 'app/model.html', {'model': report})
    else:
        return redirect(login)


def accuracy(request):
    if 'login' in request.session and request.session['login']:
        return render(request, 'app/accuracy.html')
    else:
        return redirect(login)


def accuracy_(request):
    f = open('media/model/history.pckl', 'rb')
    history = pickle.load(f)
    f.close()
    fig, ax = plt.subplots()
    ax.plot(history['accuracy'], color='g')
    ax.legend(['train', 'val'], loc='upper left')
    ax.set(xlabel='epoch', ylabel='accuracy', title='model accuracy')
    ax.grid()
    response = HttpResponse(content_type='image/jpg')
    canvas = FigureCanvasAgg(fig)
    canvas.print_jpg(response)
    return response


def upload(request):
    try:
        if request.method == 'POST':
            file_name = datetime.datetime.now().strftime('%d%m%y%I%M%S')
            user_image = 'media/temp/' + file_name + '.jpg'
            flower = read_csv("media/flower_info.csv")
            pr = open('media/model/prediction.pckl', 'rb')
            if request.FILES:
                image_file = request.FILES['user_image']
                with open(user_image, 'wb') as fw:
                    fw.write(image_file.read())
            else:
                datauri = request.POST.get("datauri")
                image_file = re.sub("^data:image/png;base64,", "", datauri)
                image_file = base64.b64decode(image_file)
                with open(user_image, 'wb') as fw:
                    fw.write(image_file)
            model_path = "D:/Models/tf2-preview_inception_v3_feature_vector_4"
            model = tf.keras.Sequential([
                hub.KerasLayer(hub.load(model_path), output_shape=[2048], trainable=False),
                tf.keras.layers.Dense(5, activation="sigmoid")
            ])
            model.build([None, 224, 224, 3])
            model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
            model.load_weights('media/model/model.h5')
            input_image = detect_object(user_image)
            img = cv2.imread(input_image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = np.array(img) / 255.0
            weights = model.predict(img)
            pred = weights.argmax(axis=1)[0]
            threshold = 0.50 if pred < 4.0 else 0.90
            if max(weights[0]) > threshold:
                flower = (flower.loc[flower['name'] == pickle.load(pr)[pred]]).values.tolist()
            else:
                raise Exception("Unknown Flower")
            pr.close()
            return render(request, 'app/upload.html', {'flower': flower[0], 'um': user_image})
        else:
            if 'login' in request.session and request.session['login']:
                return render(request, 'app/upload.html')
            else:
                return redirect(login)
    except Exception as ex:
        return render(request, 'app/upload.html', {'message': ex})


def detect_object(user_image):
    file_name = datetime.datetime.now().strftime('%d%m%y%I%M%S')
    output_image = 'media/temp/' + file_name + '.jpg'
    img = cv2.imread(user_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    mask = 255 - mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    mask = (2 * (mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    cv2.imwrite(output_image, result)
    return output_image


def login(request):
    try:
        if request.method == 'POST':
            username = str(request.POST.get("username")).strip()
            password = str(request.POST.get("password")).strip()
            if username == app.admin.username and password == app.admin.password:
                request.session['login'] = True
                return redirect(about)
            else:
                message = 'Invalid username or password'
                return render(request, 'app/login.html', {'message': message})
        else:
            request.session['login'] = False
            return render(request, 'app/login.html')
    except Exception as ex:
        return render(request, 'app/login.html', {'message': ex})
