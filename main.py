from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import cv2
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import A4
import io
from typing import List

app = FastAPI()

@app.post("/scan")
async def scan(files: List[UploadFile] = File(...)):

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)

    elements = []

    for file in files:
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

        _, img_buffer = cv2.imencode(".jpg", scan)

        img_stream = io.BytesIO(img_buffer.tobytes())
        img = Image(img_stream)
        img.drawHeight = A4[1]
        img.drawWidth = A4[0]

        elements.append(img)

    doc.build(elements)

    return Response(
        content=pdf_buffer.getvalue(),
        media_type="application/pdf"
    )
