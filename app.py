from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
from main import process_video
import threading
import json
import time
from typing import Optional

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# In-memory storage for progress tracking (in production, use Redis or database)
processing_tasks = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['video']
    
    if file.filename == '' or file.filename is None:
        flash('No file selected')
        return redirect(request.url)
    
    # At this point, we know file.filename is not None
    filename = secure_filename(file.filename)
    
    if file and allowed_file(filename):
        # Generate unique filename to avoid conflicts
        unique_filename = str(uuid.uuid4()) + os.path.splitext(filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Generate output filename
        output_filename = 'output_' + unique_filename
        
        # Store task info
        processing_tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'input_file': filepath,
            'output_file': output_filename,  # Store the output filename
            'error': None
        }
        
        # Start processing in background thread
        thread = threading.Thread(target=process_video_background, args=(task_id, filepath, unique_filename))
        thread.start()
        
        # Redirect to progress page
        return redirect(url_for('processing', task_id=task_id))
    else:
        flash('Invalid file format. Please upload MP4, AVI, MOV, or MKV files.')
        return redirect(url_for('index'))

def process_video_background(task_id, input_filepath, unique_filename):
    """Process video in background thread"""
    try:
        output_filename = 'output_' + unique_filename
        output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        # Update task info
        processing_tasks[task_id]['output_file'] = output_filename
        
        # Call the actual processing function with task_id for progress tracking
        success = process_video(input_filepath, output_filepath, task_id)
        
        if success:
            processing_tasks[task_id]['status'] = 'completed'
            processing_tasks[task_id]['progress'] = 100
        else:
            processing_tasks[task_id]['status'] = 'failed'
            processing_tasks[task_id]['error'] = 'Processing failed'
    except Exception as e:
        processing_tasks[task_id]['status'] = 'failed'
        processing_tasks[task_id]['error'] = str(e)
        print(f"Error processing video: {str(e)}")

@app.route('/processing/<task_id>')
def processing(task_id):
    if task_id not in processing_tasks:
        flash('Invalid task ID')
        return redirect(url_for('index'))
    return render_template('processing.html', task_id=task_id)

@app.route('/progress/<task_id>')
def progress(task_id):
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = processing_tasks[task_id]
    return jsonify({
        'status': task['status'],
        'progress': task['progress'],
        'output_file': task['output_file'],
        'error': task.get('error')
    })

@app.route('/results/<filename>')
def results(filename):
    return render_template('results.html', filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)

@app.route('/processed/<filename>')
def processed_video(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
