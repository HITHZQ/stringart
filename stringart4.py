import numpy as np
import cv2
from skimage.draw import line_aa
import math
import time
import os

class BenchmarkGridGenerator:
    def __init__(self, target_img_path, unit_size=200):
        self.unit_size = unit_size
        
        # 1. 读取并预处理图片
        img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError("无法读取图片")
        
        # 调整到单元格大小，作为"标准答案"
        self.target_img = cv2.resize(img, (unit_size, unit_size))
        # 误差计算用的反色图 (255=黑, 0=白)
        self.target_grid = 255.0 - self.target_img.astype(float)
        
    def generate_single_art(self, num_nails, max_lines, line_weight):
        """核心生成逻辑：返回 (图像, 耗时, 误差)"""
        
        start_time = time.time()
        
        # --- A. 预计算 ---
        nails = []
        center = (self.unit_size / 2, self.unit_size / 2)
        radius = self.unit_size / 2 - 2
        for i in range(num_nails):
            angle = 2 * np.pi * i / num_nails
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            nails.append((np.clip(x, 0, self.unit_size-1), np.clip(y, 0, self.unit_size-1)))

        line_cache = {}
        for i in range(num_nails):
            for j in range(i + 1, num_nails):
                p1, p2 = nails[i], nails[j]
                rr, cc, val = line_aa(p1[1], p1[0], p2[1], p2[0])
                valid = (rr >= 0) & (rr < self.unit_size) & (cc >= 0) & (cc < self.unit_size)
                line_cache[(i, j)] = (rr[valid], cc[valid], val[valid])
                line_cache[(j, i)] = (rr[valid], cc[valid], val[valid])

        # --- B. 绘图循环 ---
        error_matrix = self.target_grid.copy()
        current_canvas = np.zeros((self.unit_size, self.unit_size), dtype=float)
        path = [0]
        current_nail = 0
        search_step = 1 if num_nails < 200 else 2 # 加速预览

        for _ in range(max_lines):
            best_nail = -1
            max_score = -1.0
            
            for next_nail in range(0, num_nails, search_step):
                if next_nail == current_nail: continue
                key = (current_nail, next_nail)
                if key not in line_cache: continue
                
                rr, cc, val = line_cache[key]
                score = np.sum(error_matrix[rr, cc] * val)
                if score > max_score:
                    max_score = score
                    best_nail = next_nail
            
            if best_nail != -1:
                path.append(best_nail)
                rr, cc, val = line_cache[(current_nail, best_nail)]
                error_matrix[rr, cc] -= line_weight * val
                error_matrix = np.clip(error_matrix, 0, 255)
                current_canvas[rr, cc] += line_weight * val
                current_nail = best_nail
            else:
                break
        
        end_time = time.time()
        duration = end_time - start_time
        
        # --- C. 后处理与误差计算 ---
        # 1. 生成最终视觉图 (白底黑线)
        final_darkness = np.clip(current_canvas, 0, 255)
        visual_img = (255.0 - final_darkness).astype(np.uint8)
        
        # 2. 计算误差 (Mean Absolute Error)
        # 比较 "标准答案(self.target_img)" 和 "生成图(visual_img)"
        diff = np.abs(self.target_img.astype(float) - visual_img.astype(float))
        mae = np.mean(diff)
        
        return visual_img, duration, mae

    def add_info_header(self, img, params, duration, error):
        """绘制双行信息头"""
        h, w = img.shape
        header_height = 50 # 头部高度
        
        # 创建白底
        new_img = np.full((h + header_height, w), 255, dtype=np.uint8)
        new_img[header_height:, :] = img 
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.35 # 字体稍微小一点以防溢出
        thickness = 1
        color = 0
        
        # 第一行: 参数
        line1 = f"N:{params[0]} L:{params[1]} W:{params[2]}"
        (tw1, th1), _ = cv2.getTextSize(line1, font, scale, thickness)
        x1 = (w - tw1) // 2
        y1 = 20
        cv2.putText(new_img, line1, (x1, y1), font, scale, color, thickness, cv2.LINE_AA)
        
        # 第二行: 时间与误差
        line2 = f"Time:{duration:.2f}s Err:{error:.1f}"
        (tw2, th2), _ = cv2.getTextSize(line2, font, scale, thickness)
        x2 = (w - tw2) // 2
        y2 = 40
        cv2.putText(new_img, line2, (x2, y2), font, scale, color, thickness, cv2.LINE_AA)
        
        # 边框
        cv2.rectangle(new_img, (0,0), (w-1, h+header_height-1), 180, 1)
        
        return new_img

    def run_grid_search(self, nails_list, lines_list, weight_list):
        total = len(nails_list) * len(lines_list) * len(weight_list)
        print(f"准备生成 {total} 张对比图...")
        
        thumbnails = []
        count = 0
        
        for n in nails_list:
            for l in lines_list:
                for w in weight_list:
                    count += 1
                    print(f"[{count}/{total}] 计算中: N={n}, L={l}, W={w} ...", end="\r")
                    
                    # 生成
                    img, duration, error = self.generate_single_art(n, l, w)
                    
                    # 标注
                    labeled_img = self.add_info_header(img, (n, l, w), duration, error)
                    thumbnails.append(labeled_img)
        
        print("\n正在拼合大图...")
        self.stitch_and_save(thumbnails)

    def stitch_and_save(self, images):
        if not images: return
        
        count = len(images)
        cols = math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)
        
        h, w = images[0].shape
        big_canvas = np.full((rows * h, cols * w), 255, dtype=np.uint8)
        
        for idx, img in enumerate(images):
            r = idx // cols
            c = idx % cols
            y, x = r * h, c * w
            big_canvas[y:y+h, x:x+w] = img
            
        save_path = "full_benchmark_result.png"
        cv2.imwrite(save_path, big_canvas)
        print(f"\n[成功] 结果已保存: {save_path}")
        
        try: os.startfile(save_path)
        except: pass

# --- 运行配置 ---
if __name__ == "__main__":
    target_path = r"C:\Users\86188\Desktop\python\20260128\1.png"
    
    # 定义测试范围
    
    # 1. 钉子 (影响几何形状和计算速度)
    nails_options = [100, 150, 200, 250, 300, 350] 
    
    # 2. 迭代/线条数 (影响黑度和覆盖率)
    lines_options = [1000, 2000, 3000, 4000, 5000, 6000]
    
    # 3. 粗细 (影响对比度)
    weight_options = [2.5, 5, 7.5, 10, 12.5, 15] 
    
    # 分辨率设为 200，平衡速度与可视性
    gen = BenchmarkGridGenerator(target_path, unit_size=600)
    
    gen.run_grid_search(nails_options, lines_options, weight_options)