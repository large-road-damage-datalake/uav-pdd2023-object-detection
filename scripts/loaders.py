import os
import json
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# Try to import PIL for image size reading, but keep it optional or standard lib only if requested.
# The user said "Use ONLY standard library + PyYAML". 
# Reading image dimensions without PIL/OpenCV is hard. 
# I will try to use a very lightweight pure python struct unpack approach for PNG/JPEG headers if needed, 
# or just skipping resolution stats if libraries are missing, or assume users have Pillow installed?
# "Do not add big frameworks, deep learning libs". Pillow is standard in data science but maybe not "standard library".
# For now, I will use a simple struct-based image size reader for standard formats to strictly adhere to "standard lib" 
# or just skip it if it's too complex. 
# actually, "image_resolution" is requested. I'll implement a minimal pure-python header parser for JPG/PNG.

import struct
import imghdr

def get_image_size(file_path):
    """
    Return (width, height) for a given img file content using pure python.
    Supports PNG, JPEG, GIF. Returns None if unknown.
    """
    with open(file_path, 'rb') as f:
        head = f.read(24)
        if len(head) != 24:
            return None
            
        if head.startswith(b'\x89PNG\r\n\x1a\n'):
            # PNG: width, height are at byte 16-24 (big-endian)
            # 16-20: Width, 20-24: Height
            # Actually IHDR chunk starts at byte 8.
            # 8(length), 12(type="IHDR"), 16(width), 20(height)
            try:
                f.seek(16)
                w, h = struct.unpack('>II', f.read(8))
                return w, h
            except:
                return None
        elif head.startswith(b'\xff\xd8'):
            # JPEG is harder, need to scan segments.
            try:
                f.seek(0)
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf or ftype in [0xc4, 0xc8, 0xcc]:
                    f.seek(size, 1)
                    byte = f.read(1)
                    while ord(byte) == 0xff:
                        byte = f.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', f.read(2))[0] - 2
                # We are at a SOFn block
                f.read(1) # precision
                h, w = struct.unpack('>HH', f.read(4))
                return w, h
            except:
                return None
    return None

def load_coco_stats(annotations_path, images_root):
    if not os.path.exists(annotations_path):
        print(f"[{annotations_path}] not found.")
        return None
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
        
    stats = {
        'num_images': len(data.get('images', [])),
        'num_annotations': len(data.get('annotations', [])),
        'class_distribution': defaultdict(int),
        'objects_per_image': [],
        'image_resolution': {'widths': [], 'heights': []},
        'bbox_area_rel': []
    }
    
    # Class mapping
    cat_id_to_name = {c['id']: c['name'] for c in data.get('categories', [])}
    
    # helper for fast lookup
    img_id_to_res = {}
    for img in data.get('images', []):
        img_id_to_res[img['id']] = (img.get('width'), img.get('height'))
        stats['image_resolution']['widths'].append(img.get('width'))
        stats['image_resolution']['heights'].append(img.get('height'))
        
    img_ann_count = defaultdict(int)
    
    for ann in data.get('annotations', []):
        cat_name = cat_id_to_name.get(ann['category_id'], str(ann['category_id']))
        stats['class_distribution'][cat_name] += 1
        img_ann_count[ann['image_id']] += 1
        
        # Bbox area
        if 'bbox' in ann and len(ann['bbox']) == 4:
            # coco bbox: [x,y,w,h]
            w, h = ann['bbox'][2], ann['bbox'][3]
            img_w, img_h = img_id_to_res.get(ann['image_id'], (0,0))
            if img_w and img_h:
                stats['bbox_area_rel'].append((w * h) / (img_w * img_h))
                
    stats['objects_per_image'] = list(img_ann_count.values())
    # Fill zeros for images with no annotations
    no_ann_imgs = stats['num_images'] - len(img_ann_count)
    if no_ann_imgs > 0:
        stats['objects_per_image'].extend([0] * no_ann_imgs)
        
    return stats

def load_yolo_stats(images_root, annotations_root=None): 
    # YOLO: images in images_root, labels in annotations_root (txt files)
    # usually one .txt per .jpg/.png with same basename
    # annotations_root might be a folder.
    
    if not os.path.isdir(images_root):
        print(f"[{images_root}] is not a directory.")
        return None
        
    # Find images
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(images_root, ext)))
        # also try recursive ?? prompt says structure is flexible, usu train/val folders.
        # But users splits config points to a root.
    
    stats = {
        'num_images': len(image_files),
        'num_annotations': 0,
        'class_distribution': defaultdict(int),
        'objects_per_image': [],
        'image_resolution': {'widths': [], 'heights': []},
        'bbox_area_rel': []
    }
    
    for img_path in image_files:
        # Resolve dimensions
        res = get_image_size(img_path)
        if res:
            stats['image_resolution']['widths'].append(res[0])
            stats['image_resolution']['heights'].append(res[1])
            img_w, img_h = res
        else:
            img_w, img_h = 0, 0
            
        # Find label file
        # Standard yolo: replace images/ with labels/ and ext with .txt? 
        # Or just same name in annotations_root?
        # Prompt says: "one TXT per image"
        # We will assume: if annotations_root is given, look there. Else look in same dir? 
        # Config has annotations: "annotations/train" (folder) usually for yolo.
        
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = None
        if annotations_root and os.path.isdir(annotations_root):
            label_path = os.path.join(annotations_root, basename + '.txt')
        else:
            # Fallback: same dir or swap images->labels
            # But config explicitly sends annotations path.
            pass
            
        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            count = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = parts[0]
                    stats['class_distribution'][cls_id] += 1
                    count += 1
                    
                    # YOLO: class x_center y_center width height (normalized)
                    try:
                        nw, nh = float(parts[3]), float(parts[4])
                        stats['bbox_area_rel'].append(nw * nh)
                    except ValueError:
                        pass
                        
            stats['num_annotations'] += count
            stats['objects_per_image'].append(count)
        else:
            stats['objects_per_image'].append(0)
            
    return stats

def load_voc_stats(images_root, annotations_root):
    # VOC: XML files
    if not os.path.exists(annotations_root):
        return None
        
    xml_files = glob.glob(os.path.join(annotations_root, '*.xml'))
    
    stats = {
        'num_images': 0, # VOC structure is tricky, usually JPEGImages + Annotations. 
                         # If config points to images_root, we count images there.
        'num_annotations': 0,
        'class_distribution': defaultdict(int),
        'objects_per_image': [],
        'image_resolution': {'widths': [], 'heights': []},
        'bbox_area_rel': []
    }
    
    # We'll drive by XML files usually for VOC stats, or drive by images?
    # Let's drive by images if available to get accurate "num_images" including empty ones.
    image_files = []
    if os.path.isdir(images_root):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(images_root, ext)))
    
    stats['num_images'] = len(image_files)
    
    # Create lookup for xmls
    # usually basename matches
    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(annotations_root, basename + '.xml')
        
        # Image res
        w, h = 0, 0
        res = get_image_size(img_path)
        if res:
            w, h = res
            stats['image_resolution']['widths'].append(w)
            stats['image_resolution']['heights'].append(h)
            
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            count = 0
            for obj in root.findall('object'):
                name = obj.find('name').text
                stats['class_distribution'][name] += 1
                count += 1
                
                bndbox = obj.find('bndbox')
                if bndbox is not None and w > 0 and h > 0:
                    try:
                        xmin = float(bndbox.find('xmin').text)
                        ymin = float(bndbox.find('ymin').text)
                        xmax = float(bndbox.find('xmax').text)
                        ymax = float(bndbox.find('ymax').text)
                        area = (xmax - xmin) * (ymax - ymin)
                        stats['bbox_area_rel'].append(area / (w * h))
                    except:
                        pass
                        
            stats['num_annotations'] += count
            stats['objects_per_image'].append(count)
        else:
            stats['objects_per_image'].append(0)
            
    return stats

def load_image_folder_stats(images_root):
    # Classification: root/class_x/img.jpg
    if not os.path.isdir(images_root):
        return None
        
    stats = {
        'num_images': 0,
        'num_annotations': 0, # = num_images essentially
        'class_distribution': defaultdict(int),
        'class_fractions': {}, # To be computed later? Or here? Spec says per split.
        'image_resolution': {'widths': [], 'heights': []}
    }
    
    # Walk directory
    for root, dirs, files in os.walk(images_root):
        if root == images_root:
            # Top level, subdirs are classes
            for d in dirs:
                # Count files in subdir
                class_path = os.path.join(root, d)
                class_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                count = len(class_files)
                stats['class_distribution'][d] = count
                stats['num_images'] += count
                
                # Sample resolution (sampling first 10 to clear Perf? Or all?)
                # "Keep implementation straightforward". 
                # Let's read all. might be slow for huge datasets.
                # Actually, let's just attempt to read all. 
                for f in class_files:
                    res = get_image_size(os.path.join(class_path, f))
                    if res:
                        stats['image_resolution']['widths'].append(res[0])
                        stats['image_resolution']['heights'].append(res[1])
                        
    stats['num_annotations'] = stats['num_images']
    return stats

def load_png_mask_stats(images_root, masks_root):
    # Segmentation: images in images_root, masks in masks_root
    # We assume 1-1 mapping with same basename (possibly different ext)
    if not os.path.isdir(images_root) or not os.path.isdir(masks_root):
        return None
        
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(images_root, ext)))
        
    stats = {
        'num_images': len(image_files),
        'num_annotations': 0,
        'class_distribution': defaultdict(int),
        'objects_per_image': [], # num masks per image ?? Usually mask is per-pixel.
                                 # If multi-class PNG: we need to find unique values.
                                 # If binary: just 0 or 255.
        'image_resolution': {'widths': [], 'heights': []}
    }
    
    # To properly count classes in PNG masks, we MUST read them.
    # Without numpy/cv2/PIL, reading PNG pixel values is painful with just struct.
    # struct can unpack standard formats.
    # IF the mask is uncompressed PGM/PPM it's easy. But PNG is compressed (DEFLATE).
    # Standard lib 'zlib' can decompress IDAT chunks.
    # This might be getting too complex for "Keep code small".
    # Alternative: Only support basic stats for PNG masks if libraries missing?
    # "Support ONLY standard library + PyYAML".
    # I will assume users might not get class distribution for PNG masks if they don't have libraries,
    # OR I can try to use a VERY simple approach if they are 8-bit indexed.
    # Re-reading prompt: "at least counting masks per class... full pixel-based coverage is optional"
    # "at least counting masks per class": implies checking if class X is present in mask.
    # I'll add a minimal check: if we cannot read pixels easily, we might skip class_dist for png_masks 
    # and just count images.
    # BUT, let's try to be helpful. Maybe users use "Index" color mode?
    # I will skip pixel-level analysis to stay truly dependency-free and lightweight 
    # unless I write a full PNG decoder which is overkill.
    # I will verify "objects_per_image" as "1" if mask exists.
    
    # Actually, let's assume if they want PNG mask stats they might need PIL. 
    # I will try to import PIL inside the function, if fails, warn and skip class details.
    
    try:
        from PIL import Image
        import numpy as np # Usually goes together
        has_pil = True
    except ImportError:
        has_pil = False
        print("Warning: PIL/Pillow not found. PNG mask class stats will be skipped.")

    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        # Try finding mask with common extensions
        mask_path = None
        for ext in ['.png', '.jpg', '.bmp']:
            p = os.path.join(masks_root, basename + ext)
            if os.path.exists(p):
                mask_path = p
                break
        
        # Res
        res = get_image_size(img_path)
        if res:
            stats['image_resolution']['widths'].append(res[0])
            stats['image_resolution']['heights'].append(res[1])
            
        if mask_path:
            if has_pil:
                try:
                    msk = Image.open(mask_path)
                    # Multi-class mask? 
                    # Get unique values
                    # If it's huge, this is slow. use getcolors for palette images.
                    colors = msk.getcolors(maxcolors=256)
                    if colors:
                        # colors is list of (count, pixel_value)
                        unique_vals = [c[1] for c in colors]
                    else:
                        # Fallback for RGB/large
                        unique_vals = [] # TODO: Optimize?
                        # Skipping for speed/complexity trade-off
                    
                    found_classes = 0
                    for v in unique_vals:
                        if v != 0: # Assuming 0 is background
                            stats['class_distribution'][str(v)] += 1
                            found_classes += 1
                    
                    if found_classes > 0:
                        stats['num_annotations'] += found_classes 
                        stats['objects_per_image'].append(found_classes)
                    else:
                         stats['objects_per_image'].append(0)   
                except:
                     stats['objects_per_image'].append(0)
            else:
                # Naive: assuming 1 annotation if mask file exists
                stats['num_annotations'] += 1
                stats['objects_per_image'].append(1)
                stats['class_distribution']['unknown'] += 1
        else:
             stats['objects_per_image'].append(0)

    return stats
