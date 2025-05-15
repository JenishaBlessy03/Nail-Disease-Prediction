from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import os
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure random key

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure database
DATABASE = 'users.db'

def init_db():
    """Initialize the database."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Load the trained DenseNet121 model
model = load_model('densenet121_model.h5')
class_labels = ['Acral_Lentiginous_Melanoma', 'Healthy Nail', 'Onychogryphosis', 'Blue Finger', 'Clubbing', 'Pitting']

def preprocess_image(img_path):
    """Preprocess the image for prediction."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Home Route (Landing Page)
@app.route('/')
def home():
    return render_template('home.html')

# User Selection Page Route
@app.route('/user')
def user():
    return render_template('user.html')

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    session.clear()
    if request.method == 'POST':
        username = request.form.get('username').strip()
        email = request.form.get('email').strip()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password or not confirm_password:
            flash('Please fill out all fields.')
            return render_template('signup.html')
        if password != confirm_password:
            flash('Passwords do not match.')
            return render_template('signup.html')

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                      (username, email, hashed_password))
            conn.commit()
            flash('Signup successful! Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or Email already exists.')
            return render_template('signup.html')
        finally:
            conn.close()

    return render_template('signup.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form.get('username_or_email').strip()
        password = request.form.get('password')

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? OR email=?", (username_or_email, username_or_email))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['last_login'] = username_or_email  # Store last login
            return redirect(url_for('prediction'))
        else:
            flash('Invalid credentials.')
            return render_template('login.html')

    return render_template('login.html')

# Logout Route
@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.')
    return redirect(url_for('login'))

# Prediction Route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user_id' not in session:
        flash('Please login to continue.')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]
            return render_template('prediction.html', filename=filename, result=predicted_class)
    
    return render_template('prediction.html')

# Precautions Route
@app.route('/precautions/<disease>')
def precautions(disease):
    disease_pages = {
        "Acral_Lentiginous_Melanoma": "melanoma.html",
        "Onychogryphosis": "onychogryphosis.html",
        "Blue Finger": "blue_finger.html",
        "Clubbing": "clubbing.html",
        "Pitting": "pitting.html"
    }
    
    if disease in disease_pages:
        return render_template(disease_pages[disease], disease=disease)
    else:
        flash("Invalid disease selected.")
        return redirect(url_for('prediction'))

if __name__ == '__main__':
    app.run(debug=True)
