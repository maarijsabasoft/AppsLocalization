import os, io, base64
import numpy as np
import cv2
import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, send_file, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont
import easyocr
import json
import uuid
import logging
import re
import time
import secrets
from authlib.integrations.flask_client import OAuth
from flask_session import Session
from authlib.common.errors import AuthlibBaseError
import jwt
from werkzeug.utils import secure_filename
import tempfile

# Production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Production Flask config
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config["UPLOAD_FOLDER"] = os.getenv('UPLOAD_FOLDER', '/tmp/private_uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv('DATABASE_URL', "sqlite:///" + os.path.join(tempfile.gettempdir(), "users.db"))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Database and session setup
db = SQLAlchemy(app)
app.config["SESSION_TYPE"] = os.getenv('SESSION_TYPE', "filesystem")
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_FILE_DIR"] = os.getenv('SESSION_DIR', '/tmp/flask_session')
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
Session(app)

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

# Global font cache - preloaded at startup
FONT_CACHE = {}
TOP_FONTS = []
DEFAULT_FONT_SIZE = 24

def get_production_fonts():
    """Production-ready font detection with Docker/container support."""
    system = platform.system()
    font_paths = []
    
    # Docker/Debian/Ubuntu fonts (most common in production)
    docker_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
    ]
    
    if system == "Windows":
        windows_fonts = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\tahoma.ttf",
            r"C:\Windows\Fonts\verdana.ttf"
        ]
        font_paths.extend(windows_fonts)
    elif system == "Darwin":  # macOS
        mac_fonts = [
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/SFNS.ttf"
        ]
        font_paths.extend(mac_fonts)
    
    font_paths.extend(docker_fonts)  # Always try Docker fonts
    
    available_fonts = []
    for path in font_paths:
        if os.path.exists(path) and os.path.isfile(path):
            try:
                test_font = ImageFont.truetype(path, 12)
                font_name = os.path.splitext(os.path.basename(path))[0].replace('-regular', '').replace('.ttc', '.ttf')
                available_fonts.append((path, font_name))
                logger.info(f"✓ Production font loaded: {font_name}")
            except Exception as e:
                logger.debug(f"✗ Failed to load font {path}: {e}")
    
    # Always ensure default font
    try:
        default_font = ImageFont.load_default()
        available_fonts.append(("default", "default"))
    except:
        pass
    
    return available_fonts[:10]  # Top 10 fonts

def preload_fonts(fonts):
    """Pre-cache fonts with multiple sizes for instant rendering."""
    global FONT_CACHE, TOP_FONTS
    TOP_FONTS = fonts
    
    cache_hits = 0
    for path, name in fonts:
        if path == "default":
            try:
                FONT_CACHE[f"{name}_default"] = ImageFont.load_default()
                cache_hits += 1
            except:
                continue
        else:
            try:
                sizes = [12, 16, 20, 24, 32, 48, 64]  # Common sizes
                for size in sizes:
                    key = f"{name}_{size}"
                    FONT_CACHE[key] = ImageFont.truetype(path, size)
                    cache_hits += 1
                logger.info(f"✓ Pre-cached {name} ({len(sizes)} sizes)")
            except Exception as e:
                logger.warning(f"Failed to preload {name}: {e}")
    
    logger.info(f"🎉 Font preloading complete: {cache_hits} variants from {len(fonts)} families")
    return len(FONT_CACHE) > 0

def get_cached_font(font_name="default", size=DEFAULT_FONT_SIZE):
    """Get closest cached font for instant rendering."""
    key = f"{font_name}_{size}"
    if key in FONT_CACHE:
        return FONT_CACHE[key]
    
    # Find closest size
    font_keys = [k for k in FONT_CACHE.keys() if font_name in k and k != f"{font_name}_default"]
    if font_keys:
        sizes = [int(k.split('_')[-1]) for k in font_keys]
        closest_size = max([s for s in sizes if s <= size], default=min(sizes))
        return FONT_CACHE[f"{font_name}_{closest_size}"]
    
    # Ultimate fallback
    return ImageFont.load_default()

# Initialize fonts at startup (before OAuth)
font_initialized = preload_fonts(get_production_fonts())

# OAuth setup (after font init)
oauth = OAuth(app)

# Google OAuth
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID', '836571438073-g4foa0u929gskfrqhbi7q7omrl7pif2t.apps.googleusercontent.com'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET', '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'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# GitHub OAuth
github = oauth.register(
    name='github',
    client_id=os.getenv('GITHUB_CLIENT_ID', 'Ov23liJ6E1lObYC6fPOB'),
    client_secret=os.getenv('GITHUB_CLIENT_SECRET', 'e259922033f1d826b9866ba05a2ef0a14dd566f8'),
    authorize_url='https://github.com/login/oauth/authorize',
    access_token_url='https://github.com/login/oauth/access_token',
    client_kwargs={'scope': 'user:email'},
    api_base_url='https://api.github.com/'
)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=True)
    auth_provider = db.Column(db.String(50), nullable=True)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

RESOLUTIONS = {
    "iphone-15": (1290, 2796), "iphone-14": (1179, 2556), "iphone-13": (1170, 2532),
    "android-fhd": (1080, 2400), "android-qhd": (1440, 3200),
    "iphone-15-land": (2796, 1290), "android-fhd-land": (2400, 1080),
    "ipad-pro": (2048, 2732), "ipad-mini": (1536, 2048), "android-tab": (1600, 2560),
    "hd": (1920, 1080), "qhd": (2560, 1440), "4k": (3840, 2160),
    "square-1to1": (1080, 1080), "portrait-3to4": (1350, 1800),
    "portrait-2to3": (1200, 1800), "landscape-4to3": (1600, 1200),
    "landscape-16to9": (1920, 1080),
}

def preprocess_image_for_ocr(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    scale_factor = 2.0
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"temp_{uuid.uuid4().hex[:8]}.png")
    cv2.imwrite(temp_path, img)
    return temp_path, scale_factor

def clean_text(text):
    text = re.sub(r'[^\x20-\x7E]', '', text).strip()
    return text if len(text) >= 2 and any(c.isalnum() for c in text) else None

def perform_ocr(path):
    start_time = time.time()
    try:
        preprocessed_path, scale_factor = preprocess_image_for_ocr(path)
        if not preprocessed_path:
            return []

        reader = easyocr.Reader(["en"], model_storage_directory="/tmp/model", gpu=False)
        result = reader.readtext(preprocessed_path, width_ths=1.0, height_ths=0.7, mag_ratio=2.0,
                               decoder="greedy", min_size=5, text_threshold=0.3, low_text=0.3, batch_size=16)

        os.remove(preprocessed_path)

        adjusted_results = []
        for box, text, confidence in result:
            if confidence >= 0.15 and (cleaned_text := clean_text(text)):
                scaled_box = [[x / scale_factor, y / scale_factor] for x, y in box]
                adjusted_results.append((scaled_box, cleaned_text))

        logger.info(f"OCR completed in {time.time() - start_time:.2f}s")
        return adjusted_results
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return []

def choose_contrasting_color(region):
    if region.size == 0:
        return (0, 0, 0, 255)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return (0, 0, 0, 255) if np.mean(gray) > 128 else (255, 255, 255, 255)

def measure_text(draw, text, font):
    try:
        if hasattr(draw, "textbbox"):
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            return r - l, b - t
        return draw.textsize(text, font=font)
    except:
        return len(text) * 10, 20  # Fallback estimation

def translate_and_replace(path, target_lang):
    start_time = time.time()
    cv_img = cv2.imread(path)
    if cv_img is None:
        return None, None

    translator = GoogleTranslator(source="auto", target=target_lang)
    boxes = perform_ocr(path)
    
    if not boxes:
        logger.warning("No text detected")
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)), []

    mask = np.zeros(cv_img.shape[:2], np.uint8)
    for box, _ in boxes:
        cv2.fillPoly(mask, [np.array(box, np.int32)], 255)

    clean = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)
    image = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    
    # Use pre-cached font
    font_name = TOP_FONTS[0][1] if TOP_FONTS else "default"
    font = get_cached_font(font_name, DEFAULT_FONT_SIZE)

    texts_list = []
    for box, text in boxes:
        try:
            trans = translator.translate(text) or text
        except:
            trans = text

        x0, y0 = int(min(p[0] for p in box)), int(min(p[1] for p in box))
        x1, y1 = int(max(p[0] for p in box)), int(max(p[1] for p in box))
        region = cv_img[y0:y1, x0:x1] if (y1 > y0 and x1 > x0) else np.zeros((10,10,3), np.uint8)
        color = choose_contrasting_color(region)
        bw, bh = x1 - x0, y1 - y0

        # Simple text fitting with cached font
        max_width = bw * 0.9
        lines = [line.strip() for line in trans.split('\n') if line.strip()]
        if not lines:
            lines = [trans]

        # Use fixed font size with scaling
        best_size = min(bh // max(1, len(lines)), 48)
        font = get_cached_font(font_name, best_size)
        
        line_height = best_size * 1.2
        pos_y = y0
        
        for line in lines:
            tw, th = measure_text(draw, line, font)
            pos_x = x0 + max(0, (bw - tw) / 2)
            
            texts_list.append({
                'text': line, 'left': pos_x, 'top': pos_y, 'fontSize': best_size,
                'fill': f'rgb({color[0]},{color[1]},{color[2]})', 'fontFamily': font_name,
                'fontWeight': 'normal', 'fontStyle': 'normal', 'stroke': None,
                'strokeWidth': 0, 'lineHeight': 1.2, 'textDecoration': '',
                'textBackgroundColor': '', 'textAlign': 'left', 'shadow': '',
                'opacity': 1.0, 'width': max_width
            })
            pos_y += line_height

    logger.info(f"Processing completed in {time.time() - start_time:.2f}s")
    return image, texts_list

def edge_avg_color(img):
    arr = np.array(img.convert("RGB"))
    b = min(10, arr.shape[0]//2, arr.shape[1]//2)
    edges = np.vstack([
        arr[:b, :, :].reshape(-1, 3),
        arr[-b:, :, :].reshape(-1, 3),
        arr[:, :b, :].reshape(-1, 3),
        arr[:, -b:, :].reshape(-1, 3)
    ])
    return tuple(int(c) for c in edges.mean(axis=0))

def pad_keep_aspect(img, target_w, target_h, pad_color):
    img_ratio = img.width / img.height
    tgt_ratio = target_w / target_h
    if img_ratio > tgt_ratio:
        new_w, new_h = target_w, int(target_w / img_ratio)
    else:
        new_h, new_w = target_h, int(target_h * img_ratio)

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", (target_w, target_h), pad_color + (255,))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    bg.paste(resized, offset)
    return bg, offset, new_w / img.width

# Routes
@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "fonts_loaded": len(FONT_CACHE),
        "timestamp": time.time()
    })

@app.route("/api/fonts")
def api_fonts():
    return jsonify([{"name": name, "available": True} for _, name in TOP_FONTS])

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
            return redirect(url_for("login"))
        
        user = User(
            username=username, email=email,
            password=generate_password_hash(password, method="pbkdf2:sha256"),
            auth_provider="local"
        )
        db.session.add(user)
        db.session.commit()
        flash("Signup successful!", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(email=request.form["email"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            login_user(user)
            return redirect(url_for("index"))
        flash("Invalid credentials", "error")
    return render_template("login.html")

@app.route("/google-login")
def google_login():
    callback_url = url_for('google_auth_callback', _external=True)
    session['google_nonce'] = secrets.token_urlsafe(16)
    return google.authorize_redirect(callback_url, nonce=session['google_nonce'])

@app.route("/auth/google/callback")
def google_auth_callback():
    try:
        token = google.authorize_access_token()
        user_info = google.parse_id_token(token, nonce=session.pop('google_nonce', None))
        
        if not user_info.get("email"):
            flash("Google login failed: Missing email", "error")
            return redirect(url_for("login"))
        
        user = User.query.filter_by(email=user_info["email"]).first()
        if not user:
            user = User(
                username=user_info.get("name", user_info["email"].split("@")[0]),
                email=user_info["email"],
                auth_provider="google"
            )
            db.session.add(user)
            db.session.commit()
        
        login_user(user)
        return redirect(url_for("index"))
    except Exception as e:
        logger.error(f"Google auth failed: {e}")
        flash("Google login failed", "error")
        return redirect(url_for("login"))

@app.route("/github-login")
def github_login():
    return github.authorize_redirect(url_for('github_auth_callback', _external=True))

@app.route("/auth/github/callback")
def github_auth_callback():
    try:
        token = github.authorize_access_token()
        user_info = github.get('user', token=token).json()
        
        # Get email
        email = user_info.get('email')
        if not email:
            emails = github.get('user/emails', token=token).json()
            email = next((e['email'] for e in emails if e['primary'] and e['verified']), 
                        f"{user_info['login']}@github.com")
        
        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(username=user_info['login'], email=email, auth_provider='github')
            db.session.add(user)
            db.session.commit()
        
        login_user(user)
        return redirect(url_for("index"))
    except Exception as e:
        logger.error(f"GitHub auth failed: {e}")
        flash("GitHub login failed", "error")
        return redirect(url_for("login"))

@app.route("/logout")
@login_required
def logout():
    user_id = current_user.id
    for f in os.listdir(app.config["UPLOAD_FOLDER"]):
        if f.startswith(f"user_{user_id}_"):
            try:
                os.remove(os.path.join(app.config["UPLOAD_FOLDER"], f))
            except:
                pass
    
    for key in ['last_image_filename', 'last_edited_filename', 'last_texts_json', 
                'last_image_width', 'last_image_height', 'last_fonts', 'google_nonce']:
        session.pop(key, None)
    
    logout_user()
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        lang = request.form.get("language", "es")
        res_key = request.form.get("resolution", "")
        file = request.files.get("image")
        
        if not file or not file.filename:
            flash("Please upload an image", "error")
            return render_template("index.html")
        
        if not secure_filename(file.filename).lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            flash("Unsupported format", "error")
            return render_template("index.html")
        
        user_id = current_user.id if current_user.is_authenticated else "guest"
        
        # Cleanup old files
        for f in os.listdir(app.config["UPLOAD_FOLDER"]):
            if f.startswith(f"user_{user_id}_"):
                try:
                    os.remove(os.path.join(app.config["UPLOAD_FOLDER"], f))
                except:
                    pass
        
        # Save and process
        filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}.png"
        in_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(in_path)
        
        clean_img, texts_list = translate_and_replace(in_path, lang)
        if clean_img is None:
            os.remove(in_path)
            flash("Processing failed", "error")
            return render_template("index.html")
        
        # Apply resolution if needed
        scale, offset = 1.0, (0, 0)
        if res_key in RESOLUTIONS:
            target_w, target_h = RESOLUTIONS[res_key]
            pad_color = edge_avg_color(clean_img)
            clean_img, offset, scale = pad_keep_aspect(clean_img, target_w, target_h, pad_color)
        
        # Scale text positions
        for t in texts_list:
            t['left'] = t['left'] * scale + offset[0]
            t['top'] = t['top'] * scale + offset[1]
            t['fontSize'] *= scale
            if 'width' in t:
                t['width'] *= scale
        
        # Save processed image
        processed_filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}_processed.png"
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)
        clean_img.save(processed_path, format="PNG")
        
        # Store in session
        session.update({
            'last_image_filename': processed_filename,
            'last_texts_json': json.dumps(texts_list),
            'last_image_width': clean_img.width,
            'last_image_height': clean_img.height,
            'last_fonts': [name for _, name in TOP_FONTS]
        })
        
        os.remove(in_path)
        
        # Encode for instant display
        with open(processed_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        
        return render_template("index.html",
                             clean_image=encoded,
                             texts_json=session['last_texts_json'],
                             fonts=session['last_fonts'],
                             image_width=clean_img.width,
                             image_height=clean_img.height,
                             success="Translation complete!")
    
    return render_template("index.html")

@app.route("/save-edited", methods=["POST"])
@login_required
def save_edited():
    user_id = current_user.id
    data = request.json
    
    try:
        data_url = data['dataURL'].split(',')[1]
        img_data = base64.b64decode(data_url)
        img = Image.open(io.BytesIO(img_data)).convert("RGBA")
        
        filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}_edited.png"
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        img.save(path, format="PNG")
        
        session['last_edited_filename'] = filename
        return {"success": True, "filename": filename}
    except Exception as e:
        logger.error(f"Save edited failed: {e}")
        return {"error": "Save failed"}, 500

@app.route("/download")
@login_required
def download():
    user_id = current_user.id
    edited = session.get('last_edited_filename')
    processed = session.get('last_image_filename')
    
    # Try edited first
    if edited:
        path = os.path.join(app.config["UPLOAD_FOLDER"], edited)
        if os.path.exists(path):
            return send_file(path, as_attachment=True, download_name="translated_edited.png")
    
    # Fallback to processed
    if processed:
        path = os.path.join(app.config["UPLOAD_FOLDER"], processed)
        if os.path.exists(path):
            return send_file(path, as_attachment=True, download_name="translated.png")
    
    flash("No image available", "error")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv('PORT', 5000)), debug=False)
