from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import cv2
import numpy as np

app = FastAPI()

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    scan = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    _, buffer = cv2.imencode(".jpg", scan)

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg"
    )
