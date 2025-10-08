
import os, io, base64
import numpy as np
import cv2
from flask import Flask, render_template, request, send_file, redirect, url_for, flash, session
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
from authlib.integrations.flask_client import OAuth
from flask_session import Session
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "supersecret"
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
db = SQLAlchemy(app)
from flask_session import Session

app.config["SESSION_TYPE"] = "filesystem"  # or "sqlalchemy"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_FILE_DIR"] = os.path.join(os.getcwd(), "flask_session")
Session(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

# OAuth setup
oauth = OAuth(app)
REDIRECT_URI = "https://appslocalization.com/auth/google/callback"  # must match Google Console


google = oauth.register(
    name='google',
    client_id='836571438073-g4foa0u929gskfrqhbi7q7omrl7pif2t.apps.googleusercontent.com',  # Replace with your Google Client ID
    client_secret='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',  # Replace with your Google Client Secret
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    refresh_token_url=None,
    client_kwargs={'scope': 'openid email profile'},
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration'
)

github = oauth.register(
    name='github',
    client_id='Ov23liJ6E1lObYC6fPOB',  # Replace with your GitHub Client ID
    client_secret='e259922033f1d826b9866ba05a2ef0a14dd566f8',  # Replace with your GitHub Client Secret
    authorize_url='https://github.com/login/oauth/authorize',
    access_token_url='https://github.com/login/oauth/access_token',
    client_kwargs={'scope': 'user:email'},
    api_base_url='https://api.github.com/'
)
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=True)  # Nullable for OAuth users
    auth_provider = db.Column(db.String(50), nullable=True)  # e.g., 'google', 'github'
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

RESOLUTIONS = {
    "iphone-15":        (1290, 2796),
    "iphone-14":        (1179, 2556),
    "iphone-13":        (1170, 2532),
    "android-fhd":      (1080, 2400),
    "android-qhd":      (1440, 3200),
    "iphone-15-land":   (2796, 1290),
    "android-fhd-land": (2400, 1080),
    "ipad-pro":         (2048, 2732),
    "ipad-mini":        (1536, 2048),
    "android-tab":      (1600, 2560),
    "hd":               (1920, 1080),
    "qhd":              (2560, 1440),
    "4k":               (3840, 2160),
    "square-1to1":      (1080, 1080),
    "portrait-3to4":    (1350, 1800),
    "portrait-2to3":    (1200, 1800),
    "landscape-4to3":   (1600, 1200),
    "landscape-16to9":  (1920, 1080),
}

def preprocess_image_for_ocr(img_path):
    """Preprocess image to enhance small text detection efficiently."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Simplify preprocessing: adjust contrast and apply lighter denoising
    img = cv2.convertScaleAbs(img, alpha=1.3, beta=0)  # Reduced contrast factor
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Use lighter denoising to save time
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Reduce scale factor for faster processing
    scale_factor = 1.5  # Reduced from 2 to 1.5
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)  # Faster interpolation
    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"temp_{uuid.uuid4().hex[:8]}.png")
    cv2.imwrite(temp_path, img)
    return temp_path, scale_factor

def clean_text(text):
    """Clean detected text to remove random characters and noise."""
    text = re.sub(r'[^\x20-\x7E]', '', text).strip()
    if len(text) < 2 or not any(c.isalnum() for c in text):
        return None
    return text

def perform_ocr(path):
    start_time = time.time()
    try:
        preprocessed_path, scale_factor = preprocess_image_for_ocr(path)
        if not preprocessed_path:
            logger.error(f"Failed to preprocess image at {path}")
            return []

        # Initialize EasyOCR with optimized parameters
        reader = easyocr.Reader(["en"], model_storage_directory="model", gpu=False)  # Disable GPU for faster initialization on CPU
        result = reader.readtext(
            preprocessed_path,
            width_ths=0.7,
            height_ths=0.7,
            mag_ratio=1.5,  # Reduced from 2.0 to 1.5 for speed
            decoder="greedy",
            min_size=5,
            text_threshold=0.3,
            batch_size=16  # Enable batch processing
        )

        try:
            os.remove(preprocessed_path)
        except:
            logger.warning(f"Failed to delete preprocessed file {preprocessed_path}")

        adjusted_results = []
        for box, text, confidence in result:
            if confidence < 0.2:
                continue
            cleaned_text = clean_text(text)
            if not cleaned_text:
                continue
            scaled_box = [[x / scale_factor, y / scale_factor] for x, y in box]
            adjusted_results.append((scaled_box, cleaned_text))

        logger.debug(f"OCR completed in {time.time() - start_time:.2f} seconds")
        return adjusted_results
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        return []

def choose_contrasting_color(region):
    if region.size == 0:
        return (0,0,0,255)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return (0,0,0,255) if np.mean(gray) > 128 else (255,255,255,255)

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
    mask = np.zeros(cv_img.shape[:2], np.uint8)

    for box, _ in boxes:
        cv2.fillPoly(mask, [np.array(box, np.int32)], 255)

    clean = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)
    image = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font_path = "static/font/arial.ttf"

    texts_list = []
    for box, text in boxes:
        if not text:
            continue

        try:
            trans = translator.translate(text)
            if not trans or len(trans.strip()) < 1:
                logger.warning(f"Translation returned empty for '{text}'")
                trans = text
        except Exception as e:
            logger.warning(f"Translation failed for '{text}': {str(e)}")
            trans = text

        x0, y0 = int(min(p[0] for p in box)), int(min(p[1] for p in box))
        x1, y1 = int(max(p[0] for p in box)), int(max(p[1] for p in box))
        region = cv_img[y0:y1, x0:x1] if (y1 > y0 and x1 > x0) else np.zeros((10,10,3), np.uint8)
        color = choose_contrasting_color(region)
        bw, bh = x1 - x0, y1 - y0

        # Set maximum width for wrapping
        max_width = bw * 0.9
        font = ImageFont.truetype(font_path, 5)  # Start small
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

        # Adjust font size for all lines
        best_size = 5
        high = max(min(bw, bh), 10)
        for size in range(5, high + 1, 2):  # Step by 2 for faster search
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

        line_height = measure_text(draw, "A", font)[1] * 1.2  # Line spacing
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
    b = 10  # Reduced edge size for faster computation
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

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered. Please login instead.", "error")
            return redirect(url_for("login"))

        hashed_pw = generate_password_hash(password, method="pbkdf2:sha256")
        new_user = User(
            username=username,
            email=email,
            password=hashed_pw,
            auth_provider="local"   # 👈 ensure local signup users are marked correctly
        )
        db.session.add(new_user)
        db.session.commit()

        flash("Signup successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        # allow local users (auth_provider=local or None for backward compatibility)
        if user and (user.auth_provider in [None, "local"]) and check_password_hash(user.password, password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("index"))

        flash("Invalid credentials", "error")
    return render_template("login.html")


@app.route("/google-login")
def google_login():
    redirect_uri = url_for('google_auth_callback', _external=True)
    return google.authorize_redirect(redirect_uri)
@app.route("/auth/google/callback")
def google_auth_callback():
    try:
        print("\n🟢 Step 1: Callback received.")
        token = google.authorize_access_token()
        print("🟢 Step 2: Token obtained successfully.")
        print(json.dumps(token, indent=2))

        try:
            user_info = google.parse_id_token(token)
            print("🟢 Step 3: User info (ID token):", user_info)
        except Exception as parse_err:
            print("⚠️ Step 3 failed - ID token parse error:", parse_err)
            user_info = None

        if not user_info:
            print("🟡 Falling back to userinfo endpoint...")
            resp = google.get("userinfo")
            print("🟢 Step 4: Response from userinfo:", resp.status_code)
            print(resp.text)
            user_info = resp.json()

        if not user_info or not user_info.get("email"):
            print("🔴 Missing email or user info:", user_info)
            flash("Google login failed. Missing email info.", "error")
            return redirect(url_for("login"))

        email = user_info.get("email")
        username = user_info.get("name", email.split("@")[0])

        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(username=username, email=email, auth_provider="google")
            db.session.add(user)
            db.session.commit()

        login_user(user)
        flash("Logged in successfully via Google!", "success")
        return redirect(url_for("index"))

    except Exception as e:
        print("🔴 GOOGLE LOGIN FAILED:", e)
        import traceback
        print(traceback.format_exc())
        flash("Google login failed. Please try again.", "error")
        return redirect(url_for("login"))

@app.route("/github-login")
def github_login():
    redirect_uri = url_for('github_auth_callback', _external=True)
    return github.authorize_redirect(redirect_uri)

@app.route("/auth/github/callback")
def github_auth_callback():
    token = github.authorize_access_token()
    resp = github.get('user', token=token)
    user_info = resp.json()
    email = user_info.get('email')
    if not email:
        resp_emails = github.get('user/emails', token=token)
        emails = resp_emails.json()
        email = next((e['email'] for e in emails if e['primary'] and e['verified']), user_info['login'] + '@github.com')
    username = user_info.get('login', email.split('@')[0])

    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(username=username, email=email, auth_provider='github')
        db.session.add(user)
        db.session.commit()

    login_user(user)
    return redirect(url_for("index"))
@app.route("/logout")
@login_required
def logout():
    user_id = current_user.id
    for f in os.listdir(app.config["UPLOAD_FOLDER"]):
        if f.startswith(f"user_{user_id}_"):
            try:
                os.remove(os.path.join(app.config["UPLOAD_FOLDER"], f))
            except:
                logger.warning(f"Failed to delete file {f}")
    session.pop('last_image_filename', None)
    session.pop('last_edited_filename', None)
    session.pop('last_texts_json', None)
    session.pop('last_image_width', None)
    session.pop('last_image_height', None)
    session.pop('last_fonts', None)
    logout_user()
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    start_time = time.time()
    if request.method == "POST":
        lang = request.form.get("language")
        res_key = request.form.get("resolution")
        file = request.files.get("image")

        if not file or not file.filename:
            flash("Please upload a valid image.", "error")
            return render_template("index.html", error="Upload an image.")

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            flash("Unsupported file format. Use PNG, JPG, or BMP.", "error")
            return render_template("index.html", error="Unsupported file format.")

        user_id = current_user.id
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

        orig_w, orig_h = clean_img.size
        scale = 1.0
        offset = (0, 0)

        if res_key in RESOLUTIONS:
            target_w, target_h = RESOLUTIONS[res_key]
            pad_color = edge_avg_color(clean_img)
            clean_img, offset, scale = pad_keep_aspect(clean_img, target_w, target_h, pad_color)

        for t in texts_list:
            t['left'] = t['left'] * scale + offset[0]
            t['top'] = t['top'] * scale + offset[1]
            t['fontSize'] *= scale
            t['strokeWidth'] = (t['strokeWidth'] if 'strokeWidth' in t else 0) * scale
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

@app.route("/save-edited", methods=["POST"])
@login_required
def save_edited():
    start_time = time.time()
    user_id = current_user.id
    data = request.json
    if not data or 'dataURL' not in data:
        return {"error": "No image data provided"}, 400

    try:
        data_url = data['dataURL']
        if ',' in data_url:
            base64_string = data_url.split(',')[1]
        else:
            base64_string = data_url
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data)).convert("RGBA")
    except Exception as e:
        logger.error(f"Failed to decode or open edited image: {str(e)}")
        return {"error": "Invalid image data"}, 400

    edited_filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}_edited.png"
    edited_path = os.path.join(app.config["UPLOAD_FOLDER"], edited_filename)
    try:
        img.save(edited_path, format="PNG")
        session['last_edited_filename'] = edited_filename
        logger.debug(f"Save edited completed in {time.time() - start_time:.2f} seconds")
        return {"success": "Edited image saved", "filename": edited_filename}
    except Exception as e:
        logger.error(f"Failed to save edited image: {str(e)}")
        return {"error": "Failed to save edited image"}, 500

@app.route("/download")
@login_required
def download_file():
    start_time = time.time()
    if 'last_edited_filename' in session:
        edited_path = os.path.join(app.config["UPLOAD_FOLDER"], session['last_edited_filename'])
        if os.path.exists(edited_path):
            logger.debug(f"Download completed in {time.time() - start_time:.2f} seconds")
            return send_file(edited_path,
                             mimetype="image/png",
                             as_attachment=True,
                             download_name="translated_edited.png")
        else:
            logger.warning(f"Edited image not found at {edited_path}")
            session.pop('last_edited_filename', None)

    if 'last_image_filename' in session:
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], session['last_image_filename'])
        if os.path.exists(processed_path):
            logger.debug(f"Download completed in {time.time() - start_time:.2f} seconds")
            return send_file(processed_path,
                             mimetype="image/png",
                             as_attachment=True,
                             download_name="translated.png")
        else:
            logger.warning(f"Processed image not found at {processed_path}")
            session.pop('last_image_filename', None)

    flash("No image available for download.", "error")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
