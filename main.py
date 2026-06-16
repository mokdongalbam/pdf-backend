from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import cv2
import numpy as np

app = FastAPI()

# 가장 기본적이고 관대한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 곳에서 접근 허용 (가장 문제 안 생김)
    allow_credentials=False, # 이거 True로 하면 * 와 충돌나서 에러남. False가 정답!
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
    try:
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
        
        # 수동으로 모든 헤더를 때려박아서 브라우저를 속임
        headers = {
            "Content-Disposition": "attachment; filename=deskewed.pdf",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
        
        return Response(content=final_pdf_bytes, media_type="application/pdf", headers=headers)
    
    except Exception as e:
        print("Error Details:", str(e))
        return JSONResponse(
            status_code=500, 
            content={"message": "서버 처리 중 오류가 발생했습니다."},
            headers={"Access-Control-Allow-Origin": "*"}
        )

# OPTIONS 메소드(사전 요청) 수동 응답
@app.options("/api/deskew")
async def options_deskew():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*"
    }
    return Response(status_code=200, headers=headers)
