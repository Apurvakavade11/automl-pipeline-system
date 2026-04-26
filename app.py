from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json
import sqlite3
import time
import threading

app = Flask(__name__)
app.secret_key = 'your-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Store training progress globally
training_progress = {}

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    
    # Create training_sessions table with correct schema
    c.execute('''CREATE TABLE IF NOT EXISTS training_sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  dataset_name TEXT,
                  problem_type TEXT,
                  best_model_name TEXT,
                  best_model_path TEXT,
                  results TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

init_db()

def get_user(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user

# Import sklearn
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not installed")

def train_models_task(user_id, filepath, target_column, problem_type):
    """Background task for training models"""
    try:
        training_progress[user_id] = {'progress': 0, 'status': 'Loading data...', 'current_model': ''}
        
        # Load data
        df = pd.read_csv(filepath)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        training_progress[user_id] = {'progress': 10, 'status': 'Preprocessing data...', 'current_model': ''}
        
        # Preprocessing
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        training_progress[user_id] = {'progress': 20, 'status': 'Splitting data...', 'current_model': ''}
        
        if problem_type == 'classification':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            models = {
                'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'Decision Tree': DecisionTreeClassifier(random_state=42)
            }
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
            
            results = {}
            total_models = len(models)
            current_model_idx = 0
            
            for name, model in models.items():
                current_model_idx += 1
                training_progress[user_id] = {
                    'progress': 20 + (current_model_idx / total_models) * 70,
                    'status': f'Training {name}...',
                    'current_model': name
                }
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = {
                    'accuracy': round(accuracy_score(y_test, y_pred), 4),
                    'precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
                    'recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
                    'f1_score': round(f1_score(y_test, y_pred, average='weighted'), 4)
                }
            
            best_model_name = max(results, key=lambda x: results[x]['accuracy'])
            best_model = models[best_model_name]
            
        else:
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42)
            }
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            results = {}
            total_models = len(models)
            current_model_idx = 0
            
            for name, model in models.items():
                current_model_idx += 1
                training_progress[user_id] = {
                    'progress': 20 + (current_model_idx / total_models) * 70,
                    'status': f'Training {name}...',
                    'current_model': name
                }
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = {
                    'r2': round(r2_score(y_test, y_pred), 4),
                    'mae': round(mean_absolute_error(y_test, y_pred), 4),
                    'mse': round(mean_squared_error(y_test, y_pred), 4)
                }
            
            best_model_name = max(results, key=lambda x: results[x]['r2'])
            best_model = models[best_model_name]
        
        training_progress[user_id] = {'progress': 95, 'status': 'Saving best model...', 'current_model': best_model_name}
        
        # Save model
        model_filename = f"best_model_{user_id}_{datetime.now().timestamp()}.joblib"
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
        joblib.dump(best_model, model_path)
        
        # Save to database
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("""INSERT INTO training_sessions 
                     (user_id, dataset_name, problem_type, best_model_name, best_model_path, results) 
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  (user_id, os.path.basename(filepath), problem_type, 
                   best_model_name, model_path, json.dumps(results)))
        conn.commit()
        conn.close()
        
        training_progress[user_id] = {
            'progress': 100, 
            'status': 'Training completed successfully!',
            'current_model': best_model_name,
            'results': results,
            'problem_type': problem_type,
            'completed': True
        }
        
    except Exception as e:
        training_progress[user_id] = {
            'progress': 0,
            'status': f'Error: {str(e)}',
            'current_model': '',
            'error': True
        }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                     (username, email, hashed_password))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = get_user(username)
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, user_id, dataset_name, problem_type, best_model_name, best_model_path, results, created_at FROM training_sessions WHERE user_id = ? ORDER BY created_at DESC", (session['user_id'],))
    sessions_data = c.fetchall()
    conn.close()
    
    sessions = []
    for s in sessions_data:
        sessions.append({
            'id': s[0],
            'dataset_name': s[2],
            'problem_type': s[3],
            'best_model_name': s[4],
            'created_at': s[7] if len(s) > 7 else 'N/A'
        })
    
    return render_template('dashboard.html', sessions=sessions, username=session.get('username'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        if not file.filename.endswith('.csv'):
            flash('Only CSV files are allowed', 'danger')
            return redirect(request.url)
        
        filename = secure_filename(f"{session['user_id']}_{datetime.now().timestamp()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        df = pd.read_csv(filepath)
        preview = df.head(10).to_html(classes='table table-striped')
        
        session['uploaded_file'] = filepath
        session['uploaded_filename'] = filename
        
        return render_template('upload.html', preview=preview, columns=df.columns.tolist(), 
                             shape=df.shape, filename=filename, username=session.get('username'))
    
    return render_template('upload.html', username=session.get('username'))

@app.route('/detect_target', methods=['POST'])
def detect_target():
    filepath = session.get('uploaded_file')
    if not filepath:
        return jsonify({'error': 'No file uploaded'}), 400
    
    try:
        df = pd.read_csv(filepath)
        target_column = request.json.get('target_column')
        
        if target_column not in df.columns:
            return jsonify({'error': 'Target column not found'}), 400
        
        target_data = df[target_column]
        
        # Detect if classification or regression
        if target_data.dtype == 'object' or len(target_data.unique()) < 10:
            problem_type = 'classification'
            unique_count = len(target_data.unique())
            if unique_count == 2:
                task = 'binary_classification'
            else:
                task = 'multi_class_classification'
            n_classes = unique_count
        else:
            problem_type = 'regression'
            task = 'regression'
            n_classes = None
        
        session['target_column'] = target_column
        session['problem_type'] = problem_type
        session['n_classes'] = n_classes
        
        return jsonify({
            'success': True,
            'problem_type': problem_type,
            'task': task,
            'n_classes': n_classes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start_training', methods=['POST'])
def start_training():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    filepath = session.get('uploaded_file')
    target_column = session.get('target_column')
    problem_type = session.get('problem_type')
    
    if not filepath or not target_column:
        return jsonify({'error': 'No dataset or target column selected'}), 400
    
    # Clear previous progress
    training_progress[user_id] = {'progress': 0, 'status': 'Starting training...', 'current_model': ''}
    
    # Start training in background
    thread = threading.Thread(target=train_models_task, args=(user_id, filepath, target_column, problem_type))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Training started'})

@app.route('/training_progress/<int:user_id>')
def get_training_progress(user_id):
    if 'user_id' not in session or session['user_id'] != user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    progress = training_progress.get(user_id, {'progress': 0, 'status': 'Not started', 'current_model': ''})
    return jsonify(progress)

@app.route('/training_status')
def training_status():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    progress = training_progress.get(user_id, {'progress': 0, 'status': 'Initializing...', 'current_model': ''})
    return jsonify(progress)

@app.route('/progress_page')
def progress_page():
    if 'user_id' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))
    
    return render_template('progress_page.html', username=session.get('username'))

@app.route('/check_training_complete')
def check_training_complete():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    progress = training_progress.get(user_id, {})
    
    if progress.get('completed'):
        # Store results in session for results page
        session['training_results'] = progress.get('results', {})
        session['problem_type'] = progress.get('problem_type', 'classification')
        session['best_model_name'] = progress.get('current_model', '')
        
        return jsonify({'complete': True, 'results': progress.get('results', {})})
    
    return jsonify({'complete': False})

@app.route('/results')
def results():
    if 'user_id' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))
    
    results = session.get('training_results', {})
    problem_type = session.get('problem_type')
    
    if not results:
        flash('No training results found. Please train a model first.', 'warning')
        return redirect(url_for('upload'))
    
    return render_template('results.html', results=results, problem_type=problem_type, username=session.get('username'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'test_file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)
        
        file = request.files['test_file']
        
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        if not file.filename.endswith('.csv'):
            flash('Only CSV files are allowed', 'danger')
            return redirect(request.url)
        
        # Get latest model
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT best_model_path FROM training_sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (session['user_id'],))
        result = c.fetchone()
        conn.close()
        
        if not result:
            flash('No trained model found. Please train a model first.', 'danger')
            return redirect(url_for('upload'))
        
        try:
            model_path = result[0]
            model = joblib.load(model_path)
            
            test_df = pd.read_csv(file)
            test_df = test_df.select_dtypes(include=[np.number])
            test_df = test_df.fillna(test_df.mean())
            
            predictions = model.predict(test_df)
            test_df['Predictions'] = predictions
            
            pred_filename = f"predictions_{session['user_id']}_{datetime.now().timestamp()}.csv"
            pred_path = os.path.join(app.config['UPLOAD_FOLDER'], pred_filename)
            test_df.to_csv(pred_path, index=False)
            
            preview = test_df.head(20).to_html(classes='table table-striped')
            
            flash('Predictions completed successfully!', 'success')
            return render_template('predict.html', preview=preview, download_file=pred_filename, username=session.get('username'))
        except Exception as e:
            flash(f'Error making predictions: {str(e)}', 'danger')
            return redirect(request.url)
    
    return render_template('predict.html', username=session.get('username'))

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        flash('File not found', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/split_data', methods=['GET', 'POST'])
def split_data():
    if 'user_id' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        if not file.filename.endswith('.csv'):
            flash('Only CSV files are allowed', 'danger')
            return redirect(request.url)
        
        try:
            from sklearn.model_selection import train_test_split
            df = pd.read_csv(file)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            train_filename = f"train_{session['user_id']}_{datetime.now().timestamp()}.csv"
            test_filename = f"test_{session['user_id']}_{datetime.now().timestamp()}.csv"
            
            train_path = os.path.join(app.config['UPLOAD_FOLDER'], train_filename)
            test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            flash('Dataset split successfully!', 'success')
            return render_template('split_data.html', train_file=train_filename, 
                                 test_file=test_filename, train_shape=train_df.shape, 
                                 test_shape=test_df.shape, username=session.get('username'))
        except Exception as e:
            flash(f'Error splitting dataset: {str(e)}', 'danger')
            return redirect(request.url)
    
    return render_template('split_data.html', username=session.get('username'))

if __name__ == '__main__':
    # Delete old database to reset schema
    if os.path.exists('users.db'):
        os.remove('users.db')
        print("Old database removed. Creating fresh database...")
    
    init_db()
    print("Database initialized successfully!")
    app.run(debug=True, port=5000, threaded=True)