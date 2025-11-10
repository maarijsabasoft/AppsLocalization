import os
import io
import base64
import numpy as np
import cv2
import json
import uuid
import logging
import re
import time
import secrets
import random
from datetime import datetime, timedelta

from flask import Flask, send_from_directory, render_template, request, send_file, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash

from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont
import easyocr
import yake
from yake import KeywordExtractor
from authlib.integrations.flask_client import OAuth
from authlib.common.errors import AuthlibBaseError
from dotenv import load_dotenv

import nltk
from nltk.corpus import wordnet

# Download NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)  # For multilingual support if needed

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


app = Flask(__name__)

# csrf = CSRFProtect(app)

app.secret_key = "supersecret"
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "private_uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_FILE_DIR"] = os.path.join(os.getcwd(), "flask_session")
Session(app)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')  # your email
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')  # app password
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')

mail = Mail(app)

# OAuth setup
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='836571438073-g4foa0u929gskfrqhbi7q7omrl7pif2t.apps.googleusercontent.com',
    client_secret='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',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)
github = oauth.register(
    name='github',
    client_id='GITHUB_CLIENT_ID',
    client_secret='GITHUB_CLIENT_SECRET',
    authorize_url='https://github.com/login/oauth/authorize',
    access_token_url='https://github.com/login/oauth/access_token',
    client_kwargs={'scope': 'user:email'},
    api_base_url='https://api.github.com/'
)


class User(UserMixin, db.Model):  # <-- Order matters: UserMixin first
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = Column(String(255), nullable=True) 
    auth_provider = db.Column(db.String(20), default="local")
    is_verified = db.Column(db.Boolean, default=False)  # Your custom field
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Optional: Override is_active to use your is_verified field
    def is_active(self):
        """Override to block unverified users from logging in"""
        return self.is_verified

# Your OTPVerification model remains unchanged

class OTPVerification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    otp = db.Column(db.String(6), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='otp_records')

class TranslationUsage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    count = db.Column(db.Integer, default=0)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

RESOLUTIONS = {
    "iphone-15": (1290, 2796),
    "iphone-14": (1179, 2556),
    "iphone-13": (1170, 2532),
    "android-fhd": (1080, 2400),
    "android-qhd": (1440, 3200),
    "samsung-s21": (1080, 2400),
    "google-pixel": (1080, 2160),
    "ipad-pro": (2048, 2732),
    "ipad-mini": (1536, 2048),
    "android-tab": (1600, 2560),
    "surface-pro": (2736, 1824),
    "macbook-13": (2560, 1600),
    "macbook-16": (3072, 1920),
    "dell-xps": (3840, 2400),
    "thinkpad": (1920, 1080),
    "hd": (1920, 1080),
    "qhd": (2560, 1440),
    "4k": (3840, 2160),
    "5k": (5120, 2880),
    "8k": (7680, 4320),
    "instagram-post": (1080, 1080),
    "instagram-story": (1080, 1920),
    "facebook-cover": (820, 312),
    "twitter-header": (1500, 500),
    "square-1to1": (1080, 1080),
    "portrait-3to4": (1350, 1800),
    "portrait-2to3": (1200, 1800),
    "landscape-4to3": (1600, 1200),
    "landscape-16to9": (1920, 1080),
}

def preprocess_image_for_ocr(img_path):
    """Preprocess image to enhance text detection without destroying image quality."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Convert to grayscale for better OCR accuracy
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Get image dimensions
    height, width = gray.shape
    max_dimension = max(height, width)
    
    # Adaptive scaling - only upscale if image is small
    if max_dimension < 1000:
        scale_factor = 2.0
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    elif max_dimension < 2000:
        scale_factor = 1.5
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    else:
        scale_factor = 1.0
    
    # Light noise reduction (less aggressive)
    gray = cv2.bilateralFilter(gray, 3, 50, 50)
    
    # Enhance contrast gently using CLAHE (better than convertScaleAbs)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Convert back to BGR for EasyOCR
    processed_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"temp_{uuid.uuid4().hex[:8]}.png")
    cv2.imwrite(temp_path, processed_bgr)
    return temp_path, scale_factor

def clean_text(text):
    """Clean detected text to remove excessive noise while preserving international characters."""
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 2 or not any(c.isalnum() or c.isalpha() for c in text):
        return None
    return text

def perform_ocr(path, languages=['en', 'fr', 'de', 'es', 'it', 'pt', 'nl']):
    start_time = time.time()
    preprocessed_path = None
    try:
        preprocessed_path, scale_factor = preprocess_image_for_ocr(path)
        if not preprocessed_path:
            logger.error(f"Failed to preprocess image at {path}")
            return []
        reader = easyocr.Reader(languages, model_storage_directory="model", gpu=False)
        
        # Use better OCR parameters for accuracy
        result = reader.readtext(
            preprocessed_path,
            width_ths=0.7,
            height_ths=0.7,
            mag_ratio=1.0,  # Reduced from 2.0 to avoid over-processing
            decoder="greedy",
            min_size=5,
            text_threshold=0.5,  # Increased for better accuracy
            low_text=0.4,  # Increased for better accuracy
            batch_size=16,
            paragraph=False,
            detail=1
        )
        try:
            if preprocessed_path and os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)
        except:
            logger.warning(f"Failed to delete preprocessed file {preprocessed_path}")
        
        adjusted_results = []
        for box, text, confidence in result:
            # Increased confidence threshold for better accuracy
            if confidence < 0.2:
                continue
            cleaned_text = clean_text(text)
            if not cleaned_text:
                continue
            # Scale bounding boxes back to original image size
            scaled_box = [[x / scale_factor, y / scale_factor] for x, y in box]
            adjusted_results.append((scaled_box, cleaned_text))
        
        logger.debug(f"OCR completed in {time.time() - start_time:.2f} seconds, found {len(adjusted_results)} text regions")
        return adjusted_results
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        # Clean up on error
        try:
            if preprocessed_path and os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)
        except:
            pass
        return []

def choose_contrasting_color(region):
    if region.size == 0:
        return (0, 0, 0, 255)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return (0, 0, 0, 255) if np.mean(gray) > 128 else (255, 255, 255, 255)

def measure_text(draw, text, font):
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    return draw.textsize(text, font=font)

def translate_and_replace(path, target_lang):
    start_time = time.time()
    logger.debug(f"Processing image at path: {path}")
    cv_img = cv2.imread(path)
    if cv_img is None:
        logger.error(f"Failed to load image at {path}")
        return None, None
    translator = GoogleTranslator(source="auto", target=target_lang)
    boxes = perform_ocr(path)
    
    if not boxes:
        logger.warning("No text detected in image")
        return None, None
    
    # Create a precise mask - only cover text regions with minimal padding
    mask = np.zeros(cv_img.shape[:2], np.uint8)
    for box, _ in boxes:
        # Use exact bounding box with minimal padding (1 pixel) to minimize blur
        box_array = np.array(box, np.int32)
        x_coords = box_array[:, 0]
        y_coords = box_array[:, 1]
        # Minimal padding - just 1 pixel to ensure text is fully covered
        padding = 1
        expanded_box = np.array([
            [max(0, x_coords.min() - padding), max(0, y_coords.min() - padding)],
            [min(cv_img.shape[1], x_coords.max() + padding), max(0, y_coords.min() - padding)],
            [min(cv_img.shape[1], x_coords.max() + padding), min(cv_img.shape[0], y_coords.max() + padding)],
            [max(0, x_coords.min() - padding), min(cv_img.shape[0], y_coords.max() + padding)]
        ], np.int32)
        cv2.fillPoly(mask, [expanded_box], 255)
    
    # Use Navier-Stokes inpainting method which preserves edges better and causes less blur
    # Use minimal radius (1 pixel) to reduce background blur
    clean = cv2.inpaint(cv_img, mask, 1, cv2.INPAINT_NS)
    image = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font_path = "static/font/arial.ttf"
    texts_list = []
    for box, text in boxes:
        if not text:
            continue
        try:
            # Log original text for debugging
            logger.debug(f"Translating: '{text}' to {target_lang}")
            
            # Translate the text
            trans = translator.translate(text)
            
            if not trans or len(trans.strip()) < 1:
                logger.warning(f"Translation returned empty for '{text}'")
                trans = text
            else:
                # Verify translation is different from original (to catch errors)
                if trans.strip().lower() == text.strip().lower():
                    logger.warning(f"Translation same as original for '{text}', might be an error")
                
                logger.debug(f"Translated '{text}' -> '{trans}'")
        except Exception as e:
            logger.warning(f"Translation failed for '{text}': {str(e)}")
            trans = text
        x0, y0 = int(min(p[0] for p in box)), int(min(p[1] for p in box))
        x1, y1 = int(max(p[0] for p in box)), int(max(p[1] for p in box))
        region = cv_img[y0:y1, x0:x1] if (y1 > y0 and x1 > x0) else np.zeros((10, 10, 3), np.uint8)
        color = choose_contrasting_color(region)
        bw, bh = x1 - x0, y1 - y0
        max_width = bw * 0.9
        font = ImageFont.truetype(font_path, 5)
        lines = []
        words = trans.split()
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            tw, th = measure_text(draw, test_line, font)
            if tw <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
        best_size = 5
        high = max(min(bw, bh), 10)
        for size in range(5, high + 1, 2):
            try:
                font = ImageFont.truetype(font_path, size)
                max_line_width = max(measure_text(draw, line, font)[0] for line in lines)
                total_height = measure_text(draw, lines[0], font)[1] * len(lines)
                if max_line_width <= max_width and total_height <= bh * 0.9:
                    best_size = size
                else:
                    break
            except Exception as e:
                logger.error(f"Failed to load font at size {size}: {str(e)}")
                break
        line_height = measure_text(draw, "A", font)[1] * 1.2
        pos_y = y0
        for line in lines:
            tw, th = measure_text(draw, line, font)
            pos_x = x0 + (bw - tw) / 2
            texts_list.append({
                'text': line,
                'left': pos_x,
                'top': pos_y,
                'fontSize': best_size,
                'fill': f'rgb({color[0]},{color[1]},{color[2]})',
                'fontFamily': 'arial',
                'fontWeight': 'normal',
                'fontStyle': 'normal',
                'stroke': None,
                'strokeWidth': 0,
                'lineHeight': 1.2,
                'textDecoration': '',
                'textBackgroundColor': '',
                'textAlign': 'left',
                'shadow': '',
                'opacity': 1.0,
                'width': max_width
            })
            pos_y += line_height
    logger.debug(f"Translation and replacement completed in {time.time() - start_time:.2f} seconds")
    return image, texts_list

def edge_avg_color(img):
    arr = np.array(img.convert("RGB"))
    b = 10
    top, bottom = arr[:b, :, :], arr[-b:, :, :]
    left, right = arr[:, :b, :], arr[:, -b:, :]
    edges = np.vstack([top.reshape(-1, 3), bottom.reshape(-1, 3),
                       left.reshape(-1, 3), right.reshape(-1, 3)])
    return tuple(int(c) for c in edges.mean(axis=0))

def pad_keep_aspect(img, target_w, target_h, pad_color):
    img_ratio = img.width / img.height
    tgt_ratio = target_w / target_h
    if img_ratio > tgt_ratio:
        new_w = target_w
        new_h = int(target_w / img_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * img_ratio)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", (target_w, target_h), pad_color + (255,))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    bg.paste(resized, offset)
    return bg, offset, new_w / img.width

def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp_email(user_email, otp):
    msg = Message(
        subject="Verify Your Email – Apps Localization",
        recipients=[user_email],
        body=f"""
Hello,

Your verification code is: {otp}

This code expires in 10 minutes.

If you didn't request this, please ignore this email.

Best regards,
Apps Localization Team
        """,
        html=f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
            <h2 style="color: #4facfe;">Email Verification</h2>
            <p>Hello,</p>
            <p>Your verification code is:</p>
            <h1 style="letter-spacing: 5px; font-size: 2rem; color: #203a43;">{otp}</h1>
            <p><strong>This code expires in 10 minutes.</strong></p>
            <p>If you didn't request this, you can safely ignore this email.</p>
            <hr>
            <small>Apps Localization © {datetime.now().year}</small>
        </div>
        """
    )
    mail.send(msg)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
            return redirect(url_for("signup"))

        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "error")
            return redirect(url_for("signup"))

        # Create unverified user
        hashed_pw = generate_password_hash(password, method="pbkdf2:sha256")
        new_user = User(
            username=username,
            email=email,
            password=hashed_pw,
            auth_provider="local",
            is_verified=False
        )
        db.session.add(new_user)
        db.session.flush()  # Get user.id

        # Generate OTP
        otp_code = generate_otp()
        expiry = datetime.utcnow() + timedelta(minutes=10)

        otp_record = OTPVerification(
            user_id=new_user.id,
            otp=otp_code,
            expires_at=expiry
        )
        db.session.add(otp_record)
        db.session.commit()

        # Send OTP
        try:
            send_otp_email(email, otp_code)
        except Exception as e:
            db.session.rollback()
            flash("Failed to send verification email. Please try again.", "error")
            return redirect(url_for("signup"))

        # Store user ID in session for verification step
        session['pending_user_id'] = new_user.id
        flash("Check your email for the verification code.", "info")
        return redirect(url_for("verify_otp"))

    return render_template("signup.html")


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    if 'pending_user_id' not in session:
        flash("No pending verification. Please sign up again.", "error")
        return redirect(url_for("signup"))

    user_id = session['pending_user_id']
    user = User.query.get(user_id)

    if not user:
        session.pop('pending_user_id', None)
        return redirect(url_for("signup"))

    if request.method == "POST":
        entered_otp = request.form["otp"].strip()

        otp_record = OTPVerification.query.filter_by(
            user_id=user_id,
            otp=entered_otp
        ).first()

        if otp_record and otp_record.expires_at > datetime.utcnow():
            user.is_verified = True
            db.session.delete(otp_record)  # One-time use
            db.session.commit()

            session.pop('pending_user_id', None)
            login_user(user)
            flash("Email verified! Welcome!", "success")
            return redirect(url_for("index"))

        else:
            flash("Invalid or expired OTP. Please try again.", "error")

    # Resend OTP
    if request.args.get("resend") == "1":
        otp_code = generate_otp()
        expiry = datetime.utcnow() + timedelta(minutes=10)

        new_otp = OTPVerification(user_id=user_id, otp=otp_code, expires_at=expiry)
        db.session.add(new_otp)
        db.session.commit()

        try:
            send_otp_email(user.email, otp_code)
            flash("New OTP sent to your email.", "info")
        except:
            flash("Failed to resend OTP.", "error")

        return redirect(url_for("verify_otp"))

    return render_template("verify_otp.html", email=user.email)


@app.route("/login", methods=["GET", "POST"])
def login():
    # If user is already logged in, redirect to logout
    if current_user.is_authenticated:
        return redirect(url_for("logout"))
    
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()

        if user and user.is_verified and check_password_hash(user.password, password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("index"))

        if user and not user.is_verified:
            flash("Please verify your email first.", "warning")
            session['pending_user_id'] = user.id
            return redirect(url_for("verify_otp"))

        flash("Invalid credentials.", "error")

    return render_template("login.html")
@app.route("/google-login")
def google_login():
    callback_url = url_for('google_auth_callback', _external=True)
    nonce = secrets.token_urlsafe(16)
    session['google_nonce'] = nonce
    logger.debug(f"Initiating Google login with callback: {callback_url}, nonce: {nonce}")
    try:
        return google.authorize_redirect(callback_url, nonce=nonce)
    except Exception as e:
        logger.error(f"Google login initiation failed: {str(e)}")
        flash(f"Google login initiation failed: {str(e)}", "error")
        return redirect(url_for("login"))

@app.route("/auth/google/callback")
def google_auth_callback():
    try:
        logger.debug(f"Received Google OAuth callback with request URL: {request.url}")
        token = google.authorize_access_token()
        logger.debug(f"Access token obtained: {json.dumps({k: v for k, v in token.items() if k != 'access_token'}, indent=2)}")
        user_info = google.parse_id_token(token, nonce=session.pop('google_nonce', None))
        logger.debug(f"User info from ID token: {user_info}")
        valid_issuers = ['https://accounts.google.com', 'accounts.google.com']
        if user_info.get('iss') not in valid_issuers:
            logger.error(f"Invalid issuer in ID token: {user_info.get('iss')}")
            flash("Google login failed: Invalid issuer in ID token.", "error")
            return redirect(url_for("login"))
        email = user_info.get("email")
        username = user_info.get("name", email.split("@")[0])
        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(username=username, email=email, auth_provider="google")
            db.session.add(user)
            db.session.commit()
            logger.debug(f"Created new user: {email}")
        login_user(user)
        flash("Logged in successfully via Google!", "success")
        logger.debug("Google login successful, redirecting to index")
        return redirect(url_for("index"))
    except Exception as e:
        logger.error(f"Google login failed: {str(e)}")
        flash(f"Google login failed: {str(e)}", "error")
        return redirect(url_for("login"))


@app.route('/google8c1160341f6a72b4.html')
def google_verification():
    return send_from_directory('.', 'google8c1160341f6a72b4.html')

@app.route("/github-login")
def github_login():
    callback_url = url_for('github_auth_callback', _external=True)
    logger.debug(f"Initiating GitHub login with callback: {callback_url}")
    try:
        return github.authorize_redirect(callback_url)
    except Exception as e:
        logger.error(f"GitHub login initiation failed: {str(e)}")
        flash(f"GitHub login initiation failed: {str(e)}", "error")
        return redirect(url_for("login"))

@app.route("/auth/github/callback")
def github_auth_callback():
    try:
        logger.debug(f"Received GitHub OAuth callback with request URL: {request.url}")
        token = github.authorize_access_token()
        logger.debug(f"Access token obtained: {json.dumps({k: v for k, v in token.items() if k != 'access_token'}, indent=2)}")
        resp = github.get('user', token=token)
        if resp.status_code != 200:
            logger.error(f"Failed to retrieve user info: {resp.status_code} {resp.text}")
            flash(f"GitHub login failed: Unable to retrieve user info (status {resp.status_code}).", "error")
            return redirect(url_for("login"))
        user_info = resp.json()
        logger.debug(f"GitHub user info: {user_info}")
        email = user_info.get('email')
        if not email:
            resp_emails = github.get('user/emails', token=token)
            if resp_emails.status_code != 200:
                logger.error(f"Failed to retrieve emails: {resp_emails.status_code} {resp_emails.text}")
                flash(f"GitHub login failed: Unable to retrieve email (status {resp_emails.status_code}).", "error")
                return redirect(url_for("login"))
            emails = resp_emails.json()
            email = next((e['email'] for e in emails if e['primary'] and e['verified']), user_info['login'] + '@github.com')
            logger.debug(f"Retrieved email from emails endpoint: {email}")
        username = user_info.get('login', email.split('@')[0])
        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(username=username, email=email, auth_provider='github')
            db.session.add(user)
            db.session.commit()
            logger.debug(f"Created new user: {email}")
        login_user(user)
        flash("Logged in successfully via GitHub!", "success")
        logger.debug("GitHub login successful, redirecting to index")
        return redirect(url_for("index"))
    except Exception as e:
        logger.error(f"GitHub login failed: {str(e)}")
        flash(f"GitHub login failed: {str(e)}", "error")
        return redirect(url_for("login"))

@app.route('/templates/<path:filename>')
def serve_template_files(filename):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    return send_from_directory(base_dir, filename)

@app.route("/logout")
def logout():
    # Clean up user files if user is authenticated
    if current_user.is_authenticated:
        user_id = current_user.id
        for f in os.listdir(app.config["UPLOAD_FOLDER"]):
            if f.startswith(f"user_{user_id}_"):
                try:
                    os.remove(os.path.join(app.config["UPLOAD_FOLDER"], f))
                except:
                    logger.warning(f"Failed to delete file {f}")
    
    # Clear all session data
    session.pop('last_image_filename', None)
    session.pop('last_edited_filename', None)
    session.pop('last_texts_json', None)
    session.pop('last_image_width', None)
    session.pop('last_image_height', None)
    session.pop('last_fonts', None)
    session.pop('google_nonce', None)
    session.pop('guest_image_generated', None)
    session.pop('pending_user_id', None)
    
    # Logout the user
    logout_user()
    
    # Return JSON response for AJAX call
    return jsonify({"success": True, "message": "You are successfully logged out."})

@app.route("/tutorial")
def landing():
    return render_template("landing.html")

@app.route("/", methods=["GET", "POST"])
def index():
    start_time = time.time()
    if request.method == "POST":
        if not current_user.is_authenticated and session.get('guest_image_generated', False):
            flash("Please log in to generate another image.", "error")
            return redirect(url_for("login"))
        lang = request.form.get("language")
        res_key = request.form.get("resolution")
        file = request.files.get("image")
        if not file or not file.filename:
            flash("Please upload a valid image.", "error")
            return render_template("index.html", error="Upload an image.")
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            flash("Unsupported file format. Use PNG, JPG, or BMP.", "error")
            return render_template("index.html", error="Unsupported file format.")
        user_id = current_user.id if current_user.is_authenticated else "guest"
        for f in os.listdir(app.config["UPLOAD_FOLDER"]):
            if f.startswith(f"user_{user_id}_"):
                try:
                    os.remove(os.path.join(app.config["UPLOAD_FOLDER"], f))
                except:
                    logger.warning(f"Failed to delete old file {f}")
        user_filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}.png"
        in_path = os.path.join(app.config["UPLOAD_FOLDER"], user_filename)
        try:
            file.save(in_path)
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {str(e)}")
            flash("Failed to save uploaded file.", "error")
            return render_template("index.html", error="Failed to save file.")
        clean_img, texts_list = translate_and_replace(in_path, lang)
        if clean_img is None or texts_list is None:
            flash("Failed to process image. Please try another file.", "error")
            try:
                os.remove(in_path)
            except:
                pass
            return render_template("index.html", error="Image processing failed.")
        if not current_user.is_authenticated:
            session['guest_image_generated'] = True
        orig_w, orig_h = clean_img.size
        scale = 1.0
        offset = (0, 0)
        if res_key in RESOLUTIONS and res_key != "original":
            target_w, target_h = RESOLUTIONS[res_key]
            pad_color = edge_avg_color(clean_img)
            clean_img, offset, scale = pad_keep_aspect(clean_img, target_w, target_h, pad_color)
        for t in texts_list:
            t['left'] = t['left'] * scale + offset[0]
            t['top'] = t['top'] * scale + offset[1]
            t['fontSize'] *= scale
            t['strokeWidth'] = (t.get('strokeWidth', 0)) * scale
            if 'width' in t:
                t['width'] *= scale
        processed_filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}_processed.png"
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)
        try:
            clean_img.save(processed_path, format="PNG")
        except Exception as e:
            logger.error(f"Failed to save processed image: {str(e)}")
            flash("Failed to save processed image.", "error")
            try:
                os.remove(in_path)
            except:
                pass
            return render_template("index.html", error="Failed to save processed image.")
        session['last_image_filename'] = processed_filename
        session['last_texts_json'] = json.dumps(texts_list)
        session['last_image_width'] = clean_img.width
        session['last_image_height'] = clean_img.height
        fonts_dir = os.path.join(app.static_folder, 'font') if app.static_folder else 'static/font'
        os.makedirs(fonts_dir, exist_ok=True)
        fonts_files = [f for f in os.listdir(fonts_dir) if f.lower().endswith('.ttf')]
        fonts = [os.path.splitext(f)[0] for f in fonts_files]
        session['last_fonts'] = fonts
        try:
            os.remove(in_path)
        except:
            logger.warning(f"Failed to delete input file {in_path}")
        try:
            with open(processed_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read processed image for encoding: {str(e)}")
            flash("Failed to load processed image.", "error")
            return render_template("index.html", error="Failed to load processed image.")
        logger.debug(f"Index POST completed in {time.time() - start_time:.2f} seconds")
        return render_template("index.html",
                              clean_image=encoded,
                              texts_json=session['last_texts_json'],
                              fonts=fonts,
                              image_width=session['last_image_width'],
                              image_height=session['last_image_height'],
                              success="Translation complete!")
    return render_template("index.html")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/api/resize-image", methods=["POST"])
def resize_image():
    start_time = time.time()
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        file = request.files['image']
        resolution = request.form.get('resolution', '1920x1080')
        user_id = current_user.id if current_user.is_authenticated else "guest"
        if not file or not file.filename:
            return jsonify({"error": "Invalid image file"}), 400
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            return jsonify({"error": "Unsupported file format. Use PNG, JPG, or BMP."}), 400
        user_filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}.png"
        in_path = os.path.join(app.config["UPLOAD_FOLDER"], user_filename)
        try:
            file.save(in_path)
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {str(e)}")
            return jsonify({"error": "Failed to save file"}), 500
        try:
            width, height = map(int, resolution.split('x'))
        except:
            try:
                os.remove(in_path)
            except:
                pass
            return jsonify({"error": "Invalid resolution format"}), 400
        img = Image.open(in_path).convert("RGBA")
        pad_color = edge_avg_color(img)
        resized_img, offset, scale = pad_keep_aspect(img, width, height, pad_color)
        processed_filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}_resized.png"
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)
        try:
            resized_img.save(processed_path, format="PNG")
        except Exception as e:
            logger.error(f"Failed to save resized image: {str(e)}")
            try:
                os.remove(in_path)
            except:
                pass
            return jsonify({"error": "Failed to save resized image"}), 500
        try:
            with open(processed_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read resized image for encoding: {str(e)}")
            try:
                os.remove(in_path)
                os.remove(processed_path)
            except:
                pass
            return jsonify({"error": "Failed to load resized image"}), 500
        session['last_image_filename'] = processed_filename
        session['last_image_width'] = width
        session['last_image_height'] = height
        try:
            os.remove(in_path)
        except:
            logger.warning(f"Failed to delete input file {in_path}")
        logger.debug(f"Resize image completed in {time.time() - start_time:.2f} seconds")
        return jsonify({
            "success": True,
            "image_data": f"data:image/png;base64,{encoded}",
            "width": width,
            "height": height
        })
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return jsonify({"error": "Failed to resize image"}), 500
@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    start_time = time.time()
    user_id = current_user.id if current_user.is_authenticated else "guest"
    try:
        text = request.form.get('text')
        image_file = request.files.get('image')
        
        # Process image if provided
        if image_file:
            if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                return jsonify({'error': 'Unsupported file format. Use PNG, JPG, or BMP.'}), 400
            user_filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}.png"
            in_path = os.path.join(app.config["UPLOAD_FOLDER"], user_filename)
            try:
                image_file.save(in_path)
                texts = perform_ocr(in_path)
                text = "\n".join([t[1] for t in texts]) if texts else ""
                os.remove(in_path)
            except Exception as e:
                logger.error(f"Failed to process image: {str(e)}")
                return jsonify({'error': f'Failed to process image: {str(e)}'}), 400
        
        # Validate input
        if not text:
            return jsonify({'error': 'No text provided or extracted from image'}), 400
        
        # Extract top 7 keywords (focus on overall text)
        kw_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=7, features=None)
        keywords = kw_extractor.extract_keywords(text)
        
        # Keep only the keyword words (ignore scores)
        keyword_list = [kw for kw, score in keywords]
        keywords_str = "\n".join(keyword_list)
        
        logger.debug(f"Keyword extraction completed in {time.time() - start_time:.2f} seconds")
        return jsonify({
            'success': True,
            'keywords': keywords_str
        })
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route("/api/extract-text", methods=["POST"])
def extract_text():
    start_time = time.time()
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        file = request.files['image']
        user_id = current_user.id if current_user.is_authenticated else "guest"
        if not file or not file.filename:
            return jsonify({"error": "Invalid image file"}), 400
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            return jsonify({"error": "Unsupported file format. Use PNG, JPG, or BMP."}), 400
        user_filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}.png"
        in_path = os.path.join(app.config["UPLOAD_FOLDER"], user_filename)
        try:
            file.save(in_path)
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {str(e)}")
            return jsonify({"error": "Failed to save file"}), 500
        texts = perform_ocr(in_path)
        try:
            os.remove(in_path)
        except:
            logger.warning(f"Failed to delete input file {in_path}")
        extracted_text = "\n".join([text for _, text in texts]) if texts else "No text detected."
        logger.debug(f"Text extraction completed in {time.time() - start_time:.2f} seconds")
        return jsonify({
            "success": True,
            "extracted_text": extracted_text
        })
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return jsonify({"error": "Failed to extract text"}), 500

@app.route("/api/translate-text", methods=["POST"])
def translate_text():
    start_time = time.time()
    try:
        data = request.get_json()
        source_text = data.get('source_text')
        target_lang = data.get('target_language')
        if not source_text or not target_lang:
            return jsonify({"error": "Source text and target language are required"}), 400
        if len(source_text) > 100:
            return jsonify({"error": "Text exceeds 100 character limit"}), 400
        user_id = current_user.id if current_user.is_authenticated else "guest"
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        usage = TranslationUsage.query.filter(
            TranslationUsage.user_id == (user_id if user_id != "guest" else None),
            TranslationUsage.timestamp >= today,
            TranslationUsage.timestamp < tomorrow
        ).first()
        if usage and usage.count >= 500:
            return jsonify({"error": "Daily translation limit of 500 reached"}), 429
        translator = GoogleTranslator(source="auto", target=target_lang)
        translated_text = translator.translate(source_text)
        if not translated_text:
            logger.warning(f"Translation failed for '{source_text}' to '{target_lang}'")
            return jsonify({"error": "Translation failed, please try again"}), 500
        if not usage:
            usage = TranslationUsage(user_id=user_id if user_id != "guest" else None, count=1)
            db.session.add(usage)
        else:
            usage.count += 1
        db.session.commit()
        logger.debug(f"Text translation completed in {time.time() - start_time:.2f} seconds")
        return jsonify({
            "success": True,
            "translated_text": translated_text,
            "usage_count": usage.count
        })
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return jsonify({"error": "Failed to translate text"}), 500

@app.route("/save-edited", methods=["POST"])
@login_required
def save_edited():
    start_time = time.time()
    user_id = current_user.id
    data = request.json
    if not data or 'dataURL' not in data:
        return jsonify({"error": "No image data provided"}), 400
    try:
        data_url = data['dataURL']
        base64_string = data_url.split(',')[1] if ',' in data_url else data_url
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data)).convert("RGBA")
    except Exception as e:
        logger.error(f"Failed to decode or open edited image: {str(e)}")
        return jsonify({"error": "Invalid image data"}), 400
    edited_filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}_edited.png"
    edited_path = os.path.join(app.config["UPLOAD_FOLDER"], edited_filename)
    try:
        img.save(edited_path, format="PNG")
        session['last_edited_filename'] = edited_filename
        logger.debug(f"Save edited completed in {time.time() - start_time:.2f} seconds")
        return jsonify({"success": "Edited image saved", "filename": edited_filename})
    except Exception as e:
        logger.error(f"Failed to save edited image: {str(e)}")
        return jsonify({"error": "Failed to save edited image"}), 500

@app.route("/download")
def download_file():
    if not current_user.is_authenticated:
        flash("You must be logged in to download the translated image.", "error")
        return redirect(url_for("login"))
    start_time = time.time()
    edited_filename = session.get('last_edited_filename')
    processed_filename = session.get('last_image_filename')
    if edited_filename:
        edited_path = os.path.join(app.config["UPLOAD_FOLDER"], edited_filename)
        if os.path.exists(edited_path):
            logger.debug(f"Download completed in {time.time() - start_time:.2f} seconds")
            return send_file(
                edited_path,
                mimetype="image/png",
                as_attachment=True,
                download_name="translated_edited.png"
            )
        else:
            flash("Edited image not found.", "error")
            session.pop('last_edited_filename', None)
    if processed_filename:
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)
        if os.path.exists(processed_path):
            logger.debug(f"Download completed in {time.time() - start_time:.2f} seconds")
            return send_file(
                processed_path,
                mimetype="image/png",
                as_attachment=True,
                download_name="translated.png"
            )
        else:
            flash("Processed image not found.", "error")
            session.pop('last_image_filename', None)
    flash("No image available for download.", "error")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
