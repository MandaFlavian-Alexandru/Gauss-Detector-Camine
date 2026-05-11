import pandas as pd # pip install pandas
import os
from PIL import Image # pip install pillow

# --- CONFIGURATION ---
# 1. Path to the CSV file you just downloaded
CSV_FILE = r"C:\Users\fmanda\Downloads\labels_firide-gauss_2026-02-09-12-43-06.csv"
# 2. Path to your IMAGES folder (needed for math)
IMAGES_DIR = r"D:\Dataset-Firide"

# 3. Output folder
OUTPUT_DIR = r"D:\labels_unzipped"
# ---------------------

def convert():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"📂 Reading CSV: {CSV_FILE}")
    
    # Read CSV. MakeSense CSV usually has columns: 
    # label, cx, cy, w, h, image_name, image_width, image_height
    # BUT sometimes it's different. We'll inspect it safely.
    try:
        df = pd.read_csv(CSV_FILE, header=None)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    count = 0
    print("🔄 Converting...")

    # Iterate through rows
    for index, row in df.iterrows():
        # MakeSense CSV format is usually:
        # 0:label, 1:x, 2:y, 3:w, 4:h, 5:filename, 6:img_w, 7:img_h
        # We need to be careful if headers exist or not.
        
        # Simple heuristic: find the filename column (ends with .jpg/.png)
        filename = None
        for item in row:
            if str(item).lower().endswith(('.jpg', '.png', '.jpeg')):
                filename = str(item)
                break
        
        if not filename:
            continue

        # Get Image Dimensions from real file (safest)
        img_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(img_path):
            continue
            
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except:
            continue

        # Extract coordinates (assuming standard MakeSense order if no headers)
        # If your CSV has headers, row[1] might be wrong. 
        # Let's assume the numbers are the first 4 integers found.
        try:
            # Finding x, y, w, h in the row
            # Usually they are at indices 1, 2, 3, 4 OR 0, 1, 2, 3 depending on label column
            # Let's assume standard: label, x, y, w, h, filename
            
            # If the first column is a string (label name), x is at 1
            x = float(row[1])
            y = float(row[2])
            w = float(row[3])
            h = float(row[4])
            
            # Convert to YOLO (0-1)
            cx = (x + w/2) / img_w
            cy = (y + h/2) / img_h
            nw = w / img_w
            nh = h / img_h
            
            # Write to file
            txt_name = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(OUTPUT_DIR, txt_name), "a") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            
            count += 1
            
        except:
            continue

    print(f"✅ Created labels for {count} boxes.")
    print(f"📂 Location: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert()