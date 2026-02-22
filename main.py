from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import io
from typing import List

app = FastAPI()

@app.post("/scan")
async def scan(files: List[UploadFile] = File(...)):

    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)

    width, height = A4

    for uploaded_file in files:
        contents = await uploaded_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scan = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        _, img_buffer = cv2.imencode(".jpg", scan)

        img_stream = io.BytesIO(img_buffer.tobytes())

        c.drawImage(
            img_stream,
            0,
            0,
            width=width,
            height=height
        )

        c.showPage()

    c.save()

    return Response(
        content=pdf_buffer.getvalue(),
        media_type="application/pdf"
    )
