"""
Vulnerable API - Security Lab
This API contains 7 intentional security vulnerabilities.
Your mission: Find and fix them all!
"""

import hashlib
import joblib
import json
import os
from datetime import datetime, timedelta
from typing import Optional

import jwt
import numpy as np
from cryptography.fernet import Fernet
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.preprocessing import OneHotEncoder

app = FastAPI(title="Astrological Sign Prediction API")
security = HTTPBearer()

# ============================================================================
# BUG #1: Hardcoded Secret Key (HIGH SEVERITY)
# Problem: Secret regenerates on every restart, invalidating all tokens
# ============================================================================
SECRET_KEY = Fernet.generate_key().decode()  # SECURITY BUG #1
ALGORITHM = "HS256"

# ============================================================================
# Data Models
# ============================================================================
class User(BaseModel):
    first_name: str
    last_name: str
    email: str
    age: int
    sex: str
    favorite_color: str
    favorite_food: str

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

# ============================================================================
# Initialize ML Model and Encoders
# ============================================================================
def load_model(path: str):
    """Load the pre-trained ML model"""
    return joblib.load(path)

def initialize_allowed_classes():
    """Define allowed categorical values"""
    return {
        'favorite_colors': ['Red', 'Blue', 'Green', 'Yellow', 'Purple'],
        'favorite_foods': ['Pizza', 'Pasta', 'Burger', 'Sushi', 'Salad', 'Ice Cream'],
        'sex': ['Male', 'Female']
    }

def create_encoder(allowed_classes):
    """Create and fit OneHotEncoder for categorical features"""
    encoder = OneHotEncoder(
        categories=[
            allowed_classes['sex'],
            allowed_classes['favorite_colors'],
            allowed_classes['favorite_foods']
        ],
        sparse_output=False
    )
    dummy_data = np.array([['Male', 'Red', 'Pizza']])
    encoder.fit(dummy_data)
    return encoder

def generate_encryption_suite():
    """Generate Fernet encryption suite"""
    encryption_key = Fernet.generate_key()
    cipher_suite = Fernet(encryption_key)
    return cipher_suite

# ============================================================================
# Initialize Application State
# ============================================================================
model = load_model("./models/model_fin2.pkl")
allowed_classes = initialize_allowed_classes()
encoder = create_encoder(allowed_classes)
cipher_suite = generate_encryption_suite()
users_data = []  # Store encrypted user data

# ============================================================================
# BUG #2: Weak Password Hashing (MEDIUM SEVERITY)
# Problem: SHA256 without salt is vulnerable to rainbow table attacks
# ============================================================================
def hash_password(password: str) -> str:
    """Hash password using SHA256 - SECURITY BUG #2"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return hash_password(plain_password) == hashed_password

# ============================================================================
# Mock User Database
# ============================================================================
USERS_DB = {
    "admin@example.com": {
        "username": "admin@example.com",
        "password": hash_password("admin123"),
        "role": "admin"
    },
    "user@example.com": {
        "username": "user@example.com",
        "password": hash_password("password"),
        "role": "user"
    }
}

# ============================================================================
# BUG #3: No Rate Limiting (MEDIUM SEVERITY)
# Problem: /token endpoint can be brute-forced without any rate limiting
# ============================================================================
@app.post("/token", response_model=TokenResponse)
async def login(credentials: LoginRequest):
    """
    Authenticate user and return JWT token
    SECURITY BUG #3: No rate limiting on authentication endpoint
    """
    user = USERS_DB.get(credentials.username)

    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # ========================================================================
    # BUG #4: Token Never Expires (HIGH SEVERITY)
    # Problem: No expiration time set for JWT tokens
    # ========================================================================
    token_data = {
        "sub": user["username"],
        "role": user["role"]
        # SECURITY BUG #4: Missing "exp" field - token never expires!
    }

    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

    return TokenResponse(access_token=token, token_type="bearer")

# ============================================================================
# Authentication Dependency
# ============================================================================
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info"""
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")

        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============================================================================
# Helper Functions
# ============================================================================
def validate_user_input(user: User):
    """Validate user input against allowed classes"""
    if user.favorite_color not in allowed_classes['favorite_colors']:
        raise HTTPException(status_code=400, detail="Invalid favorite color")
    if user.favorite_food not in allowed_classes['favorite_foods']:
        raise HTTPException(status_code=400, detail="Invalid favorite food")
    if user.sex not in allowed_classes['sex']:
        raise HTTPException(status_code=400, detail="Invalid sex")

def encrypt_user_data(user: User):
    """Encrypt sensitive user data"""
    return {
        "encrypted_first_name": cipher_suite.encrypt(user.first_name.encode()),
        "encrypted_last_name": cipher_suite.encrypt(user.last_name.encode()),
        "encrypted_email": cipher_suite.encrypt(user.email.encode())
    }

def preprocess_user_data(user: User):
    """Preprocess user data for ML model"""
    categorical_features = np.array([[user.sex, user.favorite_color, user.favorite_food]])
    encoded_features = encoder.transform(categorical_features).flatten()
    features = np.concatenate(([user.age], encoded_features))
    return features.reshape(1, -1)

def pseudonymize_data(encrypted_data):
    """Create pseudonymized hash of encrypted data"""
    return hashlib.sha256(encrypted_data).hexdigest()

# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    """Welcome page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Astrological Sign Prediction API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            h1 { text-align: center; }
            .info {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            code {
                background: rgba(0,0,0,0.3);
                padding: 2px 6px;
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <h1>🔮 Astrological Sign Prediction API</h1>
        <div class="info">
            <h2>Welcome!</h2>
            <p>This API predicts your astrological sign based on your profile.</p>
            <h3>Getting Started:</h3>
            <ol>
                <li>Get a token from <code>POST /token</code></li>
                <li>Use token to access <code>POST /predict</code></li>
            </ol>
            <p>📚 View API documentation at <a href="/docs" style="color: #ffd700;">/docs</a></p>
        </div>
    </body>
    </html>
    """
    return html_content

@app.post("/predict")
async def predict_sign(user: User, current_user: dict = Depends(get_current_user)):
    """
    Predict astrological sign based on user data
    Requires authentication
    """
    validate_user_input(user)
    encrypted_data = encrypt_user_data(user)
    features = preprocess_user_data(user)
    prediction = model.predict(features)[0]

    users_data.append(encrypted_data)

    return {
        "astrological_sign": prediction,
        "encrypted_first_name": encrypted_data["encrypted_first_name"].decode(),
        "encrypted_last_name": encrypted_data["encrypted_last_name"].decode(),
        "encrypted_email": encrypted_data["encrypted_email"].decode()
    }

@app.get("/pseudonymize")
async def pseudonymize_user_data(current_user: dict = Depends(get_current_user)):
    """
    Get pseudonymized user data
    Requires authentication
    """
    pseudonymized_users = [
        {
            "pseudonymized_first_name": pseudonymize_data(user["encrypted_first_name"]),
            "pseudonymized_last_name": pseudonymize_data(user["encrypted_last_name"]),
            "pseudonymized_email": pseudonymize_data(user["encrypted_email"]),
        }
        for user in users_data
    ]
    return pseudonymized_users

# ============================================================================
# BUG #5: Authentication Bypass (HIGH SEVERITY)
# Problem: Admin endpoint accessible without checking user role
# ============================================================================
@app.get("/admin/users")
async def get_all_users(current_user: dict = Depends(get_current_user)):
    """
    Get all users - Admin only
    SECURITY BUG #5: No role check! Any authenticated user can access
    """
    # Missing: if current_user.get("role") != "admin": raise HTTPException(403)

    return {
        "users": list(USERS_DB.keys()),
        "total": len(USERS_DB)
    }

# ============================================================================
# BUG #6: Insecure Direct Object Reference (MEDIUM SEVERITY)
# Problem: User can decrypt ANY user's data by guessing IDs
# ============================================================================
@app.get("/decrypt/{user_id}")
async def decrypt_data(user_id: int, current_user: dict = Depends(get_current_user)):
    """
    Decrypt user data by ID
    SECURITY BUG #6: No authorization check - users can access others' data
    """
    if user_id < 0 or user_id >= len(users_data):
        raise HTTPException(status_code=404, detail="User not found")

    # Missing: Check if current_user owns this data

    user = users_data[user_id]
    return {
        "user_id": user_id,
        "decrypted_first_name": cipher_suite.decrypt(user["encrypted_first_name"]).decode(),
        "decrypted_last_name": cipher_suite.decrypt(user["encrypted_last_name"]).decode(),
        "decrypted_email": cipher_suite.decrypt(user["encrypted_email"]).decode()
    }

# ============================================================================
# BUG #7: Missing Input Validation (MEDIUM SEVERITY)
# Problem: Age field not validated, can cause model errors or attacks
# ============================================================================
@app.post("/predict-no-auth")
async def predict_sign_no_auth(user: User):
    """
    Public prediction endpoint (for testing)
    SECURITY BUG #7: No input validation on age field
    """
    # Missing: age validation (e.g., 0 < age < 150)
    # Could accept negative ages or unrealistic values

    validate_user_input(user)  # Only validates categorical fields
    encrypted_data = encrypt_user_data(user)
    features = preprocess_user_data(user)
    prediction = model.predict(features)[0]

    return {
        "astrological_sign": prediction,
        "message": "Public prediction - no encryption saved"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# ============================================================================
# Run Application
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Vulnerable API on http://0.0.0.0:8000")
    print("📚 API Documentation: http://0.0.0.0:8000/docs")
    print("⚠️  This API contains 7 security vulnerabilities - find them all!")
    uvicorn.run("vulnerable_api:app", host="0.0.0.0", port=8000, reload=True)
