import os
import json
import random
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# Configuration
MP3_FOLDER = os.path.join('static', 'mp3')
TAGS_FILE = 'tags.json'

# Ensure tags file exists
if not os.path.exists(TAGS_FILE):
    with open(TAGS_FILE, 'w') as f:
        json.dump({}, f)

def get_mp3_files():
    """Get all MP3 files from the directory"""
    if not os.path.exists(MP3_FOLDER):
        os.makedirs(MP3_FOLDER)
    return [f for f in os.listdir(MP3_FOLDER) if f.lower().endswith('.mp3') and f[0] != '.']

def get_tags():
    """Load tags from JSON file"""
    with open(TAGS_FILE, 'r') as f:
        return json.load(f)

def save_tags(tags_data):
    """Save tags to JSON file"""
    with open(TAGS_FILE, 'w') as f:
        json.dump(tags_data, f, indent=4)

def get_untagged_mp3s():
    """Get list of MP3s that haven't been tagged yet"""
    all_mp3s = get_mp3_files()
    if not all_mp3s:
        return []

    tags_data = get_tags()

    # Check which MP3s don't have tags yet
    untagged = []
    for mp3 in all_mp3s:
        mp3_key = mp3.rsplit('.', 1)[0]  # Remove .mp3 extension
        if mp3_key not in tags_data or not tags_data[mp3_key]:
            untagged.append(mp3)

    return untagged

def get_random_mp3():
    """Get a random MP3 file that hasn't been tagged yet"""
    untagged_mp3s = get_untagged_mp3s()

    if not untagged_mp3s:
        return None  # All MP3s have been tagged

    return random.choice(untagged_mp3s)

@app.route('/')
def index():
    """Main page"""
    mp3_files = get_mp3_files()
    if not mp3_files:
        return render_template('index.html', error="No MP3 files found in the static/mp3 folder")

    random_mp3 = get_random_mp3()

    # Check if all MP3s have been tagged
    if random_mp3 is None:
        return render_template('index.html', all_done=True)

    return render_template('index.html', mp3=random_mp3)

@app.route('/random-mp3')
def random_mp3():
    """API endpoint to get a random MP3"""
    mp3 = get_random_mp3()
    if mp3:
        return jsonify({'file': mp3})
    return jsonify({'all_done': True}), 200  # Changed from error to success with all_done flag

@app.route('/save-tags', methods=['POST'])
def save_mp3_tags():
    """API endpoint to save tags for an MP3"""
    data = request.json
    mp3_file = data.get('mp3')
    tags = data.get('tags', [])

    # Remove .mp3 extension for storage
    mp3_key = mp3_file.rsplit('.', 1)[0] if mp3_file.lower().endswith('.mp3') else mp3_file

    # Load existing tags
    tags_data = get_tags()

    # Update tags for this MP3
    tags_data[mp3_key] = tags

    # Save back to file
    save_tags(tags_data)

    return jsonify({'success': True})

@app.route('/get-tags/<mp3_file>')
def get_mp3_tags(mp3_file):
    """API endpoint to get tags for a specific MP3"""
    mp3_key = mp3_file.rsplit('.', 1)[0] if mp3_file.lower().endswith('.mp3') else mp3_file
    tags_data = get_tags()
    return jsonify({'tags': tags_data.get(mp3_key, [])})

if __name__ == '__main__':
    app.run(debug=True)
