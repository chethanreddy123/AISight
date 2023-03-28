

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras

model = keras.models.load_model('all.h5')

# app.py
from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/file')
async def _file_upload(my_file: UploadFile = File(...)):

    contents = await my_file.read() 
    # print(contents)
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.resize(img, (224, 224))
    img_array = np.asarray(image)
    CheckArray = np.array([np.array(img_array)])
    Scores = model.predict(CheckArray)

    Index = ["Normal" , "Cataract" , "Diabetes" , "Glaucoma"]
    numbers = [float(i) for i in Scores[0]]
    # Calculate the sum of all numbers
    total = sum(numbers)

    # Calculate the percentage of each number
    percentage_list = [round((number / total) * 100, 2) for number in numbers]

    print(percentage_list)
    result = {"Index" : list(Index) , "Scores" : percentage_list}
    print(result)
    return result