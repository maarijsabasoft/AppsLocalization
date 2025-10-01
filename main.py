import os, io, base64
import numpy as np
import cv2
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont
import easyocr

app = Flask(__name__)
app.secret_key = "supers  ecret"
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

last_image = None

reader = easyocr.Reader(["en"], model_storage_directory="model")

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

def perform_ocr(path):
    result = reader.readtext(path, width_ths=0.8, decoder="wordbeamsearch")
    return [(b[0], b[1]) for b in result if b[2] > 0.4]

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
    translator = GoogleTranslator(source="auto", target=target_lang)
    boxes = perform_ocr(path)
    cv_img = cv2.imread(path)
    mask = np.zeros(cv_img.shape[:2], np.uint8)

    for box, _ in boxes:
        cv2.fillPoly(mask, [np.array(box, np.int32)], 255)

    clean = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)
    image = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font_path = "arial.ttf"

    for box, text in boxes:
        trans = translator.translate(text)
        x0, y0 = int(min(p[0] for p in box)), int(min(p[1] for p in box))
        x1, y1 = int(max(p[0] for p in box)), int(max(p[1] for p in box))
        region = cv_img[y0:y1, x0:x1]
        color = choose_contrasting_color(region)
        bw, bh = x1 - x0, y1 - y0

        size = 10
        font = ImageFont.truetype(font_path, size)
        while True:
            ftmp = ImageFont.truetype(font_path, size)
            tw, th = measure_text(draw, trans, ftmp)
            if tw > bw or th > bh:
                break
            size += 2
            font = ftmp

        tw, th = measure_text(draw, trans, font)
        draw.text((x0 + (bw - tw) / 2, y0 + (bh - th) / 2),
                  trans, fill=color, font=font)

    return image

def edge_avg_color(img):
    arr = np.array(img.convert("RGB"))
    b = 20
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

    resized = img.resize((new_w, new_h), Image.LANCZOS)
    bg = Image.new("RGB", (target_w, target_h), pad_color)
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    bg.paste(resized, offset)
    return bg

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
        new_user = User(username=username, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        flash("Signup successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]; password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user); return redirect(url_for("index"))
        flash("Invalid credentials", "error")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    global last_image
    if request.method == "POST":
        lang = request.form.get("language")
        res_key = request.form.get("resolution")
        file = request.files.get("image")

        if not file:
            return render_template("index.html", error="Upload an image.")

        for f in os.listdir(app.config["UPLOAD_FOLDER"]):
            try:
                os.remove(os.path.join(app.config["UPLOAD_FOLDER"], f))
            except:
                pass

        filename = "latest.png"
        in_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(in_path)

        out_img = translate_and_replace(in_path, lang)

        if res_key and res_key in RESOLUTIONS:
            target_w, target_h = RESOLUTIONS[res_key]
            pad_color = edge_avg_color(out_img)

            src_is_portrait = out_img.height > out_img.width
            tgt_is_portrait = target_h > target_w

            if src_is_portrait and not tgt_is_portrait:
                out_img = pad_keep_aspect(out_img, target_w, target_h, pad_color)
            elif (not src_is_portrait) and tgt_is_portrait:
                out_img = pad_keep_aspect(out_img, target_w, target_h, pad_color)
            else:
                out_img = out_img.resize((target_w, target_h), Image.LANCZOS)

        last_image = out_img

        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

        return render_template("index.html",
                               translated_image=encoded,
                               success="Translation complete!")

    return render_template("index.html")

@app.route("/download")
@login_required
def download_file():
    global last_image
    if last_image is None:
        return "No image available", 404

    buf = io.BytesIO()
    last_image.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png",
                     as_attachment=True,
                     download_name="translated.png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


# import os, io, base64
# import numpy as np
# import cv2
# from flask import Flask, render_template, request, send_file, redirect, url_for, flash
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
# from werkzeug.security import generate_password_hash, check_password_hash
# from flask_sqlalchemy import SQLAlchemy
# from deep_translator import GoogleTranslator
# from PIL import Image, ImageDraw, ImageFont
# import easyocr

# # --- Flask setup ---
# app = Flask(__name__)
# app.secret_key = "supers  ecret"# change in production
# app.config["UPLOAD_FOLDER"] = "uploads"
# os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# # DB setup
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
# db = SQLAlchemy(app)

# # Login setup
# login_manager = LoginManager()
# login_manager.login_view = "login"
# login_manager.init_app(app)

# # User model
# class User(UserMixin, db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(150), unique=True, nullable=False)
#     email = db.Column(db.String(150), unique=True, nullable=False)
#     password = db.Column(db.String(256), nullable=False)

# with app.app_context():
#     db.create_all()

# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))

# # OCR + Translation setup
# reader = easyocr.Reader(["en"], model_storage_directory="model")
# last_image = None

# RESOLUTIONS = {
#     "iphone-15": (1290, 2796),
#     "android-fhd": (1080, 2400),
#     "ipad-pro": (2048, 2732),
#     "hd": (1920, 1080),
#     "qhd": (2560, 1440),
#     "4k": (3840, 2160),
# }

# def perform_ocr(path):
#     result = reader.readtext(path, width_ths=0.8, decoder="wordbeamsearch")
#     return [(b[0], b[1]) for b in result if b[2] > 0.4]

# def translate_and_replace(path, target_lang):
#     translator = GoogleTranslator(source="auto", target=target_lang)
#     boxes = perform_ocr(path)
#     cv_img = cv2.imread(path)
#     mask = np.zeros(cv_img.shape[:2], np.uint8)

#     for box, _ in boxes:
#         cv2.fillPoly(mask, [np.array(box, np.int32)], 255)

#     clean = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)
#     image = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)).convert("RGBA")
#     draw = ImageDraw.Draw(image)
#     font_path = "arial.ttf"

#     for box, text in boxes:
#         trans = translator.translate(text)
#         x0, y0 = int(min(p[0] for p in box)), int(min(p[1] for p in box))
#         x1, y1 = int(max(p[0] for p in box)), int(max(p[1] for p in box))
#         region = cv_img[y0:y1, x0:x1]
#         color = (0,0,0,255) if np.mean(region) > 128 else (255,255,255,255)
#         bw, bh = x1 - x0, y1 - y0
#         size, font = 10, ImageFont.truetype(font_path, 10)
#         while True:
#             ftmp = ImageFont.truetype(font_path, size)
#             tw, th = draw.textsize(trans, font=ftmp)
#             if tw > bw or th > bh: break
#             size += 2; font = ftmp
#         draw.text((x0+(bw-tw)//2, y0+(bh-th)//2), trans, fill=color, font=font)
#     return image

# # --- Auth routes ---
# @app.route("/signup", methods=["GET", "POST"])
# def signup():
#     if request.method == "POST":
#         username = request.form["username"]
#         email = request.form["email"]
#         password = request.form["password"]

#         # ✅ Check if email already exists
#         existing_user = User.query.filter_by(email=email).first()
#         if existing_user:
#             flash("Email already registered. Please login instead.", "error")
#             return redirect(url_for("login"))

#         hashed_pw = generate_password_hash(password, method="pbkdf2:sha256")
#         new_user = User(username=username, email=email, password=hashed_pw)
#         db.session.add(new_user)
#         db.session.commit()

#         flash("Signup successful! Please log in.", "success")
#         return redirect(url_for("login"))

#     return render_template("signup.html")


# @app.route("/login", methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         email = request.form["email"]; password = request.form["password"]
#         user = User.query.filter_by(email=email).first()
#         if user and check_password_hash(user.password, password):
#             login_user(user); return redirect(url_for("index"))
#         flash("Invalid credentials", "error")
#     return render_template("login.html")

# @app.route("/logout")
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for("login"))

# # --- Tool route ---
# @app.route("/", methods=["GET", "POST"])
# @login_required
# def index():
#     global last_image
#     if request.method == "POST":
#         lang = request.form.get("language")
#         res_key = request.form.get("resolution")
#         file = request.files.get("image")
#         if not file:
#             return render_template("index.html", error="Upload an image.")
#         for f in os.listdir(app.config["UPLOAD_FOLDER"]):
#             try: os.remove(os.path.join(app.config["UPLOAD_FOLDER"], f))
#             except: pass
#         in_path = os.path.join(app.config["UPLOAD_FOLDER"], "latest.png")
#         file.save(in_path)
#         out_img = translate_and_replace(in_path, lang)
#         if res_key in RESOLUTIONS:
#             w, h = RESOLUTIONS[res_key]
#             out_img = out_img.resize((w, h), Image.LANCZOS)
#         last_image = out_img
#         buf = io.BytesIO(); out_img.save(buf, format="PNG")
#         encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
#         return render_template("index.html", translated_image=encoded, success="Translation complete!")
#     return render_template("index.html")

# @app.route("/download")
# @login_required
# def download_file():
#     global last_image
#     if not last_image: return "No image available", 404
#     buf = io.BytesIO(); last_image.save(buf, format="PNG"); buf.seek(0)
#     return send_file(buf, mimetype="image/png", as_attachment=True, download_name="translated.png")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
