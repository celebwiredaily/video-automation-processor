#!/usr/bin/env python3
import subprocess
import cv2
import mediapipe as mp
import os
import json
import sys
from pathlib import Path

# Configuration (will be overwritten by GitHub Actions)
SEARCH_QUERY = "vintage Hollywood 1960s"
NUM_IMAGES = 100
BLUR_THRESHOLD = 100  # Laplacian variance threshold for watermarks
OUTPUT_DIR = "filtered_images"
CLIP_DURATION = 3  # Seconds per image

def download_images():
    """Download images using pinterest-dl"""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING {NUM_IMAGES} IMAGES")
    print(f"Search Query: {SEARCH_QUERY}")
    print(f"{'='*60}\n")
    
    try:
        subprocess.run([
            "pinterest-dl", 
            "-q", SEARCH_QUERY,
            "-l", str(NUM_IMAGES),
            "-o", "raw_downloads",
            "--delay", "2"  # 2-second delay between requests (avoid rate limits)
        ], check=True)
        
        downloaded = len(list(Path("raw_downloads").glob("*.*")))
        print(f"\n✅ Downloaded {downloaded} images")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pinterest download failed: {e}")
        sys.exit(1)

def detect_faces(image_path):
    """Check if image contains faces using MediaPipe"""
    mp_face_detection = mp.solutions.face_detection
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        
        return results.detections is not None

def detect_watermark(image_path):
    """Detect watermarks in bottom 15% using Laplacian variance"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True  # Skip if can't read
    
    height = image.shape[0]
    bottom_region = image[int(height * 0.85):, :]  # Bottom 15%
    
    # Calculate edge density (high variance = sharp edges = likely watermark)
    laplacian_var = cv2.Laplacian(bottom_region, cv2.CV_64F).var()
    
    # High variance indicates watermark
    return laplacian_var > BLUR_THRESHOLD

def filter_images():
    """Apply face detection and watermark filtering"""
    print(f"\n{'='*60}")
    print(f"FILTERING IMAGES")
    print(f"{'='*60}\n")
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Get all downloaded images
    raw_images = list(Path("raw_downloads").glob("*.jpg")) + \
                 list(Path("raw_downloads").glob("*.png")) + \
                 list(Path("raw_downloads").glob("*.jpeg"))
    
    print(f"Total images to process: {len(raw_images)}\n")
    
    filtered_count = 0
    stats = {
        "no_face": 0,
        "watermark": 0,
        "passed": 0
    }
    
    for img_path in raw_images:
        print(f"Processing: {img_path.name}")
        
        # Filter 1: Must have faces
        if not detect_faces(str(img_path)):
            print(f"  ❌ No faces detected")
            stats["no_face"] += 1
            continue
        
        # Filter 2: No watermarks
        if detect_watermark(str(img_path)):
            print(f"  ❌ Watermark detected")
            stats["watermark"] += 1
            continue
        
        # Copy to filtered directory
        output_path = Path(OUTPUT_DIR) / f"{filtered_count:03d}.jpg"
        subprocess.run(["cp", str(img_path), str(output_path)], check=True)
        filtered_count += 1
        stats["passed"] += 1
        print(f"  ✅ Passed filters → {output_path.name}")
    
    print(f"\n{'='*60}")
    print(f"FILTERING RESULTS")
    print(f"{'='*60}")
    print(f"Total processed: {len(raw_images)}")
    print(f"No faces: {stats['no_face']}")
    print(f"Watermarks: {stats['watermark']}")
    print(f"Passed: {stats['passed']}")
    print(f"Success rate: {stats['passed']/len(raw_images)*100:.1f}%")
    print(f"{'='*60}\n")
    
    if filtered_count < 50:
        print(f"❌ ERROR: Only {filtered_count} images passed filters")
        print(f"Need at least 50 images for a 10-minute video")
        sys.exit(1)
    
    return filtered_count

def create_editly_config(num_images):
    """Generate Editly JSON config with transitions"""
    print(f"\n{'='*60}")
    print(f"CREATING VIDEO CONFIG")
    print(f"{'='*60}\n")
    
    clips = []
    
    for i in range(num_images):
        clips.append({
            "duration": CLIP_DURATION,
            "transition": {
                "name": "random",  # Random transitions (fade, slide, zoom)
                "duration": 0.5
            },
            "layers": [{
                "type": "image",
                "path": f"filtered_images/{i:03d}.jpg",
                "resizeMode": "cover"  # Fill frame, crop if needed
            }]
        })
    
    config = {
        "outPath": "output.mp4",
        "width": 1080,
        "height": 1920,  # Vertical video (TikTok/Reels/Shorts)
        "fps": 30,
        "audioFilePath": "voiceover.mp3",
        "clips": clips,
        "keepSourceAudio": False,
        "allowRemoteRequests": False
    }
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    total_duration = num_images * (CLIP_DURATION + 0.5)
    print(f"✅ Created Editly config")
    print(f"   Clips: {num_images}")
    print(f"   Estimated duration: {total_duration/60:.1f} minutes")
    print(f"   Output: 1080x1920 @ 30fps\n")

def main():
    """Main workflow controller"""
    if len(sys.argv) < 2:
        print("Usage: python main.py [download|filter|config|all]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "download":
        download_images()
    
    elif command == "filter":
        filter_images()
    
    elif command == "config":
        num_filtered = len(list(Path(OUTPUT_DIR).glob("*.jpg")))
        create_editly_config(num_filtered)
    
    elif command == "all":
        download_images()
        num_filtered = filter_images()
        create_editly_config(num_filtered)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
