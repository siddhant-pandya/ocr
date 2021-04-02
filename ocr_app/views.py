from django.shortcuts import render

from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.templatetags.static import static
from django.contrib.staticfiles.finders import find
import os

from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2

def func(model_arg, image_arg):
    model = load_model(model_arg)

    image = cv2.imread("media/"+image_arg)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    chars = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            if tW > tH:
                thresh = imutils.resize(thresh, width=32)

            else:
                thresh = imutils.resize(thresh, height=32)

            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            chars.append((padded, (x, y, w, h)))

    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    preds = model.predict(chars)

    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in labelNames]

    output_str=""

    for (pred, (x, y, w, h)) in zip(preds, boxes):
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]

        output_str = output_str + label
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    #cv2.imshow("Image", image)
    cv2.imwrite("static/ocr_app/saved.png", image)

    print("Recognised Text: ",output_str)
    context = {'txt':output_str, 'img':"static/ocr_app/saved.png"}
    return context


def index(request):
    context={}
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        
        #context = {recognised text, image}
        context = func("static/ocr_app/handwriting.model", filename)

    return render(request, 'ocr_app/index.html', context)