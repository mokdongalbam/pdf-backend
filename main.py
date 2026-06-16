from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import cv2
import numpy as np

app = FastAPI()

# 깃허브 페이지에서 접근할 수 있도록 허락해주는 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_skew_angle(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 <= angle <= 45:
                angles.append(angle)
                
    if not angles:
        return 0.0
    return np.median(angles)

@app.post("/api/deskew")
async def process_pdf(file: UploadFile = File(...)):
    input_bytes = await file.read()
    pdf = fitz.open("pdf", input_bytes)
    out_pdf = fitz.open()

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        angle = calculate_skew_angle(img_array)

        if abs(angle) > 0.1:
            (h, w) = img_array.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            rotated = cv2.warpAffine(img_array, M, (new_w, new_h), 
                                     flags=cv2.INTER_CUBIC, 
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(255, 255, 255))
        else:
            rotated = img_array

        _, buffer = cv2.imencode('.jpg', rotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img_doc = fitz.open("jpeg", buffer.tobytes())
        pdfbytes = img_doc.convert_to_pdf()
        img_pdf = fitz.open("pdf", pdfbytes)
        out_pdf.insert_pdf(img_pdf)

    final_pdf_bytes = out_pdf.write()
    
    return Response(
        content=final_pdf_bytes, 
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=deskewed.pdf"}
    )
