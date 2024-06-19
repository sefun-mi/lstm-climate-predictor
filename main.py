from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import numpy as np
import pandas as pd
from keras import models as ker_models
from sklearn.preprocessing import MinMaxScaler
from typing import List,Dict, Annotated
import models
import random
import string
from database import engine, SessionLocal
from sqlalchemy.orm import Session

app = FastAPI()
models.Base.metadata.create_all(bind=engine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

sessions: Dict[str, str] = {}

model = ker_models.load_model('lstm_model.h5')

class AuthResponse(BaseModel):
    token: str

class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    phonenum: str
    username: str
    password: str

class PredictionRequest(BaseModel):
    year: str

class PredictionResponse(BaseModel):
    prediction: models.PredictionEntity
    class Config:
        arbitrary_types_allowed = True 

class prediction_response_list(BaseModel):
    prediction_year: str
    median_tasmax: str
    mean_tas: str
    median_cwd: str
    mean_pr: str
    mean_tasmin: str
    median_hurs: str
    median_tas: str
    median_tasmin: str
    median_cdd: str
    mean_tasmax: str

class Multiple_predictions_response(BaseModel):
    predictions: List[prediction_response_list]    

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

# Utility function to make predictions
def predict(model, year):
    
    file_path = "merged_climate_data.csv"
    data = pd.read_csv(file_path)
    data = data.sort_values('code')
    data['year'] = data['code']
    data.set_index('code', inplace=True)

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    # Appending the new year as part of the sequence
    new_year_scaled = scaler.transform([[0] * (data.shape[1] - 1) + [year]])[0]
    input_seq = np.vstack([scaled_data, new_year_scaled])

    num_timesteps = 5

    if input_seq.shape[0] < num_timesteps:
        raise ValueError(f"Input sequence is too short. Expected at least {num_timesteps} timesteps but got {input_seq.shape[0]}.")

    input_seq = input_seq[-num_timesteps:]

    if input_seq.ndim == 2:
        input_seq = np.expand_dims(input_seq, axis=0)
            
    # Make the prediction
    predicted_scaled = model.predict(input_seq)
    prediction = scaler.inverse_transform(predicted_scaled)
    return prediction.tolist()

def generate_token():
    return ''.join(random.choices(string.digits, k=8))

# Dependency to get current user based on token
def get_current_user(token: str = Depends(OAuth2PasswordBearer(tokenUrl="/login"))):
    if token in sessions:
        return sessions[token]
    else:
        raise HTTPException(status_code=401, detail="Invalid token")

# Define endpoints
@app.post("/predict", response_model=prediction_response_list)
async def predict_with_model(request: PredictionRequest, db: db_dependency, username: str=Depends(get_current_user)):
    user = db.query(models.UserEntity).filter(models.UserEntity.username==username).first()
    print(user.username)
    prediction = predict(model, request.year)
    print(prediction)
    db_prediction = models.PredictionEntity(user_id=user.id, median_tasmax=prediction[0][0], mean_tas=prediction[0][1], median_cwd=prediction[0][2], mean_pr=prediction[0][3], mean_tasmin=prediction[0][4], median_hurs=prediction[0][5], median_tas=prediction[0][6], median_tasmin=prediction[0][7], median_cdd=prediction[0][8], mean_tasmax=prediction[0][9], prediction_year=request.year)

    db.add(db_prediction)
    db.commit()
    return prediction_response_list(median_tasmax=str(prediction[0][0]), mean_tas=str(prediction[0][1]), median_cwd=str(prediction[0][2]), mean_pr=str(prediction[0][3]), mean_tasmin=str(prediction[0][4]), median_hurs=str(prediction[0][5]), median_tas=str(prediction[0][6]), median_tasmin=str(prediction[0][7]), median_cdd=str(prediction[0][8]), mean_tasmax=str(prediction[0][9]), prediction_year=str(prediction[0][10]))

@app.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest, db : db_dependency):
    user = db.query(models.UserEntity).filter(models.UserEntity.username==request.username).first()
    if user.password==request.password:
        token = generate_token()
        sessions[token] = request.username
        return AuthResponse(token=token)
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/signup")
async def signup(request: SignupRequest, db : db_dependency):
    try:
        existing_user = db.query(models.UserEntity).filter(models.UserEntity.username == request.username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        new_user = models.UserEntity(username=request.username, password=request.password, phonenum=request.phonenum)
        db.add(new_user)
        db.commit()
        return {"message": "User created successfully"}
    except Exception as e:
        db.rollback()  # Rollback the transaction in case of any exception
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")
    
@app.get("/get-predictions", response_model=Multiple_predictions_response)
async def get_past_predictions(db : db_dependency):
    predictions_list = db.query(models.PredictionEntity).all()
    response_list = []
    for i in predictions_list:
        new_pred = prediction_response_list(median_tasmax=i.median_tasmax, mean_tas=i.mean_tas, median_cwd=i.median_cwd, mean_pr=i.mean_pr, mean_tasmin=i.mean_tasmin, median_hurs=i.median_hurs, median_tas=i.median_tas, median_tasmin=i.median_tasmin, median_cdd=i.median_cdd, mean_tasmax=i.mean_tasmax, prediction_year = i.prediction_year if i.prediction_year else "not found") 

        response_list.append(new_pred)
    return Multiple_predictions_response(predictions=response_list)