from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import sqlite3
from io import BytesIO
from PIL import Image
import torch
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MTCNN (face detection) and InceptionResnetV1 (face recognition)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)  # Detect one face
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize SQLite database with faces and students tables
def init_db():
    conn = sqlite3.connect('face_embeddings.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  name TEXT NOT NULL, 
                  embedding BLOB NOT NULL,
                  image BLOB NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS students 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  name TEXT NOT NULL UNIQUE, 
                  attendance TEXT)''')  # attendance stored as comma-separated timestamps
    conn.commit()
    conn.close()

# Convert uploaded file to numpy array
def file_to_image(file: UploadFile) -> np.ndarray:
    image_bytes = file.file.read()
    return np.array(Image.open(BytesIO(image_bytes)))

# Convert numpy array to bytes for SQLite storage
def image_to_bytes(image: np.ndarray) -> bytes:
    success, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Failed to encode image")
    return encoded_image.tobytes()

# Get face embedding using facenet-pytorch
def get_face_embedding(image: np.ndarray) -> np.ndarray:
    try:
        img_pil = Image.fromarray(image)
        face, prob = mtcnn(img_pil, return_prob=True)
        if face is None or prob < 0.95:  # Threshold for detection confidence
            raise ValueError("No face detected or low confidence")
        return resnet(face.unsqueeze(0)).detach().cpu().numpy()[0]
    except Exception as e:
        raise ValueError(f"Failed to generate embedding: {str(e)}")

# Endpoint to store face embedding and image
@app.post("/store_face")
async def store_face(name: str, files: List[UploadFile] = File(...)):
    try:
        if len(files) != 3:
            raise HTTPException(status_code=400, detail="Exactly 3 images are required")
        embeddings, image_bytes_list = [], []
        for file in files:
            image = file_to_image(file)
            embedding = get_face_embedding(image)
            image_bytes = image_to_bytes(image)
            embeddings.append(embedding)
            image_bytes_list.append(image_bytes)
        if not embeddings:
            raise HTTPException(status_code=400, detail="No face detected in any image")
        mean_embedding = np.mean(embeddings, axis=0)
        conn = sqlite3.connect('face_embeddings.db')
        c = conn.cursor()
        c.execute("INSERT INTO faces (name, embedding, image) VALUES (?, ?, ?)", 
                 (name, mean_embedding.tobytes(), image_bytes_list[0]))
        c.execute("INSERT OR IGNORE INTO students (name, attendance) VALUES (?, ?)", (name, ""))
        conn.commit()
        conn.close()
        return {"message": "Mean face embedding stored successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/predict_face")
async def predict_face(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith('.jpg'):
            raise HTTPException(status_code=400, detail="Only .jpg files are supported")
        image = file_to_image(file)
        unknown_embedding = get_face_embedding(image)
        conn = sqlite3.connect('face_embeddings.db')
        c = conn.cursor()
        c.execute("SELECT name, embedding FROM faces")
        results = c.fetchall()
        conn.close()
        if not results:
            raise HTTPException(status_code=400, detail="No faces in database")
        min_distance, matched_name = float('inf'), None
        for name, embedding_bytes in results:
            stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            distance = np.linalg.norm(stored_embedding - unknown_embedding)
            if distance < min_distance and distance < 1.1:
                min_distance = distance
                matched_name = name
        if matched_name:
            confidence = round((1 - min_distance / 1.1) * 100, 2)
            conn = sqlite3.connect('face_embeddings.db')
            c = conn.cursor()
            c.execute("SELECT attendance FROM students WHERE name = ?", (matched_name,))
            result = c.fetchone()
            current_attendance = result[0] if result else ""
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_attendance = f"{current_attendance},{timestamp}" if current_attendance else timestamp
            c.execute("UPDATE students SET attendance = ? WHERE name = ?", (new_attendance, matched_name))
            conn.commit()
            conn.close()
            return {"name": matched_name, "confidence": min(confidence, 100)}
        else:
            raise HTTPException(status_code=404, detail="No match found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/get_attendance")
async def get_attendance(name: str):
    try:
        conn = sqlite3.connect('face_embeddings.db')
        c = conn.cursor()
        # Debug: Print all students
        c.execute("SELECT name FROM students")
        all_students = c.fetchall()
        print(f"All students in database: {all_students}")
        print(f"Looking for student: {name}")
        # Original query
        c.execute("SELECT attendance FROM students WHERE name = ?", (name,))
        result = c.fetchone()
        if result is None:
            # Check if the student is in the faces table
            c.execute("SELECT name FROM faces WHERE name = ?", (name,))
            face_result = c.fetchone()
            if face_result:
                # Add to students table if in faces table
                print(f"Student {name} found in faces but not in students, adding to the students table.")
                c.execute("INSERT INTO students (name, attendance) VALUES (?, ?)", (name, ""))
                conn.commit()
                result = ("",)  # Empty attendance for the newly added student
            else:
                raise HTTPException(status_code=404, detail="Student not found")
        attendance_list = result[0].split(",") if result[0] else []
        conn.close()
        return {"name": name, "attendance": attendance_list}
    except HTTPException as e:
        raise e
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    init_db()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)