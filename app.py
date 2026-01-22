# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import os
import hashlib
from PIL import Image
import io
from collections import defaultdict
import subprocess
import threading
from tkinter import Tk, filedialog
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.fftpack import dct

import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 全局状态
image_cache = {
    'file_data': {},  # path -> {hashes, histogram}
    'is_indexed': False,
    'total_images': 0,
    'search_dir': ''
}

index_progress = {
    'running': False,
    'stop_requested': False,
    'current': 0,
    'total': 0,
    'current_file': '',
    'done': False
}

progress_lock = threading.Lock()

# ============ 多种哈希算法 ============

def get_ahash(img, size=16):
    """平均哈希 - 16x16分辨率"""
    try:
        img = img.convert('L').resize((size, size), Image.Resampling.LANCZOS)
        pixels = np.array(img)
        avg = pixels.mean()
        return ''.join('1' if p > avg else '0' for p in pixels.flatten())
    except:
        return None

def get_dhash(img, size=16):
    """差异哈希 - 比较相邻像素，对缩放更稳定"""
    try:
        img = img.convert('L').resize((size + 1, size), Image.Resampling.LANCZOS)
        pixels = np.array(img)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return ''.join('1' if b else '0' for b in diff.flatten())
    except:
        return None

def get_phash(img, size=32, hash_size=8):
    """感知哈希 - 使用快速DCT"""
    try:
        img = img.convert('L').resize((size, size), Image.Resampling.LANCZOS)
        pixels = np.array(img, dtype=np.float64)

        # 使用scipy快速DCT
        dct_result = dct(dct(pixels, axis=0), axis=1)

        # 取左上角低频部分
        dct_low = dct_result[:hash_size, :hash_size]
        avg = dct_low.mean()
        return ''.join('1' if v > avg else '0' for v in dct_low.flatten())
    except:
        return None

def get_color_histogram(img, bins=8):
    """颜色直方图 - 比较颜色分布"""
    try:
        img = img.convert('RGB').resize((64, 64), Image.Resampling.LANCZOS)
        pixels = np.array(img)

        # 分别计算RGB三个通道的直方图
        hist_r = np.histogram(pixels[:,:,0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(pixels[:,:,1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(pixels[:,:,2], bins=bins, range=(0, 256))[0]

        # 归一化
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(float)
        hist = hist / (hist.sum() + 1e-7)
        return hist.tolist()
    except:
        return None

# ============ 相似度计算 ============

def hamming_distance(hash1, hash2):
    """计算汉明距离"""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return float('inf')
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def hash_similarity(hash1, hash2):
    """计算哈希相似度 (0-100%)"""
    if not hash1 or not hash2:
        return 0
    dist = hamming_distance(hash1, hash2)
    max_dist = len(hash1)
    return max(0, (1 - dist / max_dist) * 100)

def histogram_similarity(hist1, hist2):
    """计算直方图相似度 (0-100%)"""
    if not hist1 or not hist2:
        return 0
    try:
        h1 = np.array(hist1)
        h2 = np.array(hist2)
        # 使用巴氏系数
        similarity = np.sum(np.sqrt(h1 * h2))
        return similarity * 100
    except:
        return 0

def calculate_overall_similarity(data1, data2):
    """综合评分 - 多算法加权平均"""
    scores = []
    weights = []

    # aHash 权重 25%
    if data1.get('ahash') and data2.get('ahash'):
        scores.append(hash_similarity(data1['ahash'], data2['ahash']))
        weights.append(25)

    # dHash 权重 30% (对缩放更稳定)
    if data1.get('dhash') and data2.get('dhash'):
        scores.append(hash_similarity(data1['dhash'], data2['dhash']))
        weights.append(30)

    # pHash 权重 30% (更准确)
    if data1.get('phash') and data2.get('phash'):
        scores.append(hash_similarity(data1['phash'], data2['phash']))
        weights.append(30)

    # 颜色直方图 权重 15%
    if data1.get('histogram') and data2.get('histogram'):
        scores.append(histogram_similarity(data1['histogram'], data2['histogram']))
        weights.append(15)

    if not scores:
        return 0

    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

# ============ 图片处理 ============

def process_single_image(file_path):
    """处理单张图片，计算所有哈希值"""
    if index_progress['stop_requested']:
        return None

    try:
        # 计算文件MD5
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        with Image.open(file_path) as img:
            data = {
                'path': file_path,
                'md5': file_hash,
                'ahash': get_ahash(img),
                'dhash': get_dhash(img),
                'phash': get_phash(img),
                'histogram': get_color_histogram(img)
            }

        with progress_lock:
            index_progress['current'] += 1
            index_progress['current_file'] = os.path.basename(file_path)

        return data
    except:
        with progress_lock:
            index_progress['current'] += 1
        return None

def scan_images_background(directory):
    global index_progress, image_cache

    index_progress['running'] = True
    index_progress['stop_requested'] = False
    index_progress['done'] = False
    index_progress['current'] = 0
    index_progress['total'] = 0
    index_progress['current_file'] = '扫描文件夹中...'

    extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    file_data = {}

    batch = []
    batch_size = 100  # 增大批次
    files_found = 0

    def process_batch(files_batch):
        with ThreadPoolExecutor(max_workers=32) as executor:  # 恢复32线程
            futures = {executor.submit(process_single_image, fp): fp for fp in files_batch}
            for future in as_completed(futures):
                if index_progress['stop_requested']:
                    return False
                result = future.result()
                if result:
                    file_data[result['path']] = result
        return True

    try:
        for root, dirs, files in os.walk(directory):
            if index_progress['stop_requested']:
                break

            for file in files:
                if index_progress['stop_requested']:
                    break
                if os.path.splitext(file)[1].lower() in extensions:
                    batch.append(os.path.join(root, file))
                    files_found += 1
                    index_progress['total'] = files_found
                    index_progress['current_file'] = f'扫描中... 已发现 {files_found} 张图片'

                    if len(batch) >= batch_size:
                        if not process_batch(batch):
                            break
                        batch = []

            if index_progress['stop_requested']:
                break

        if batch and not index_progress['stop_requested']:
            process_batch(batch)
    except Exception as e:
        index_progress['current_file'] = f'错误: {str(e)}'

    if not index_progress['stop_requested']:
        image_cache['file_data'] = file_data
        image_cache['is_indexed'] = True
        image_cache['total_images'] = len(file_data)
        image_cache['search_dir'] = directory

    index_progress['running'] = False
    index_progress['done'] = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/select-folder', methods=['POST'])
def select_folder():
    try:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title='选择图片文件夹')
        root.destroy()
        if folder:
            return jsonify({'success': True, 'folder': folder})
        return jsonify({'success': False, 'error': '已取消'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/index', methods=['POST'])
def start_index():
    if index_progress['running']:
        return jsonify({'success': False, 'error': '正在索引中'})

    data = request.get_json() or {}
    folder = data.get('folder', 'D:\\L')

    if not os.path.isdir(folder):
        return jsonify({'success': False, 'error': '无效文件夹'})

    thread = threading.Thread(target=scan_images_background, args=(folder,))
    thread.start()
    return jsonify({'success': True, 'message': '开始索引', 'folder': folder})

@app.route('/api/stop', methods=['POST'])
def stop_index():
    index_progress['stop_requested'] = True
    return jsonify({'success': True, 'message': '正在停止...'})

@app.route('/api/clear', methods=['POST'])
def clear_cache():
    global image_cache
    image_cache = {
        'file_data': {},
        'is_indexed': False,
        'total_images': 0,
        'search_dir': ''
    }
    return jsonify({'success': True, 'message': '缓存已清理'})

@app.route('/api/progress')
def get_progress():
    return jsonify({
        'running': index_progress['running'],
        'current': index_progress['current'],
        'total': index_progress['total'],
        'current_file': index_progress['current_file'],
        'done': index_progress['done'],
        'percent': round(index_progress['current'] / max(index_progress['total'], 1) * 100, 1)
    })

@app.route('/api/search', methods=['POST'])
def search_image():
    try:
        if not image_cache['is_indexed']:
            return jsonify({'success': False, 'error': '请先索引'}), 400

        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '没有图片'}), 400

        # 获取阈值参数
        threshold = float(request.form.get('threshold', 60))

        file = request.files['image']
        image_data = file.read()

        # 计算上传图片的特征
        file_md5 = hashlib.md5(image_data).hexdigest()

        with Image.open(io.BytesIO(image_data)) as img:
            search_data = {
                'md5': file_md5,
                'ahash': get_ahash(img),
                'dhash': get_dhash(img),
                'phash': get_phash(img),
                'histogram': get_color_histogram(img)
            }

        results = []
        exact_matches = []

        # 遍历所有已索引的图片计算相似度
        for path, data in image_cache['file_data'].items():
            # 完全相同 (MD5匹配)
            if data['md5'] == file_md5:
                exact_matches.append({
                    'path': path,
                    'similarity': 100.0,
                    'type': 'exact'
                })
                continue

            # 计算综合相似度
            similarity = calculate_overall_similarity(search_data, data)

            if similarity >= threshold:
                results.append({
                    'path': path,
                    'similarity': round(similarity, 1),
                    'type': 'similar'
                })

        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)

        # 限制返回数量
        results = results[:100]

        return jsonify({
            'success': True,
            'exact_matches': exact_matches,
            'similar_matches': results,
            'total_matches': len(exact_matches) + len(results),
            'threshold': threshold
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/preview-image', methods=['POST'])
def preview_image():
    """返回图片的 base64 预览"""
    try:
        data = request.get_json()
        file_path = data.get('path')

        if not file_path or not os.path.exists(file_path):
            return jsonify({'success': False, 'error': '文件不存在'}), 400

        # 读取并缩放图片以加快传输
        with Image.open(file_path) as img:
            # 保持比例缩放到最大 800px
            max_size = 800
            ratio = min(max_size / img.width, max_size / img.height, 1.0)
            if ratio < 1.0:
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # 转换为 RGB（处理 RGBA、P 模式等）
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # 保存为 JPEG base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/open-folder', methods=['POST'])
def open_folder():
    try:
        data = request.get_json()
        file_path = data.get('path')
        if file_path and os.path.exists(file_path):
            # 规范化路径，确保使用 Windows 格式的反斜杠
            normalized_path = os.path.normpath(file_path)
            # 使用 Popen 避免阻塞，并正确处理 Windows 路径
            subprocess.Popen(f'explorer /select,"{normalized_path}"', shell=True)
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': '文件不存在'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    return jsonify({
        'is_indexed': image_cache['is_indexed'],
        'total_images': image_cache['total_images'],
        'search_dir': image_cache['search_dir']
    })

if __name__ == '__main__':
    import webbrowser
    print('=' * 50)
    print('图片匹配工具 v2.0 - 全面升级版')
    print('- 多算法匹配: aHash + dHash + pHash + 颜色直方图')
    print('- 汉明距离相似度计算')
    print('- 可调节相似度阈值')
    print('http://localhost:5000')
    print('=' * 50)
    webbrowser.open('http://localhost:5000')
    app.run(debug=False, host='127.0.0.1', port=5000)
