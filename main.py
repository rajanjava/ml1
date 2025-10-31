from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and columns
df = None
model = None
x_cols = []
y_col = None
  
@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    global df
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

    return {"columns": df.columns.tolist()}

@app.post("/train/")
async def train(x_columns: list[str] = Form(...), y_column: str = Form(...)):
    global model, x_cols, y_col, df
    if df is None:
        raise HTTPException(status_code=400, detail="No CSV uploaded yet!")

    X = df[x_columns]
    y = df[y_column]

    model = LinearRegression()
    model.fit(X, y)

    x_cols = x_columns
    y_col = y_column

    return {"message": f"Model trained with X={x_columns} and Y={y_column}"}

@app.post("/predict/")
async def predict(inputs: list[float] = Form(...)):
    global model, x_cols
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet!")

    # Ensure correct number of inputs
    if len(inputs) != len(x_cols):
        raise HTTPException(status_code=400, detail=f"Expected {len(x_cols)} inputs for X={x_cols}")

    prediction = model.predict([inputs])
    return {"prediction": prediction.tolist()}
