from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os

from authentication import authenticate

app = FastAPI(title="Fingerprint Authentication API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("experiments/authentication_outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="experiments/authentication_outputs"), name="outputs")

@app.get("/")
def root():
    return {"status": "Fingerprint Authentication API running"}


@app.get("/app")
def frontend():
    return FileResponse("frontend/index.html")


@app.post("/authenticate")
async def authenticate_fingerprint(
    identity_code: str = Form(...),
    fingerprint: UploadFile = File(...),
):
    try:
        image_bytes = await fingerprint.read()

        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid fingerprint image.",
            )

        mysql_password = os.environ.get("MYSQL_PASSWORD", "")

        result = authenticate(
            identity_code=identity_code,
            fingerprint_image=image,
            mysql_password=mysql_password,
        )
        
        original_url = None
        rectified_url = None
        
        if result.get("saved_original_path"):
            filename = os.path.basename(result["saved_original_path"])
            original_url = f"/outputs/{filename}"
            
        if result.get("saved_rectified_path"):
            filename = os.path.basename(result["saved_rectified_path"])
            rectified_url = f"/outputs/{filename}"

        return {
            "authenticated": result["authenticated"],
            "identity_code": result["identity_code"],
            "distortion_probability": result["distortion_probability"],
            "is_distorted": result["is_distorted"],
            "rectification_applied": result["rectification_applied"],
            "best_score": result["best_score"],
            "best_raw_score": result["best_raw_score"],
            "best_impression_id": result["best_impression_id"],
            "scores": result.get("scores", []),
            "original_image_url": original_url,
            "rectified_image_url": rectified_url,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
