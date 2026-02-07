import numpy as np
import cv2
from skimage.draw import line_aa
import matplotlib.pyplot as plt
import math

class EvolutionStringArt:
    def __init__(self, target_img_path, num_nails=200, max_lines=4000, img_size=400):
        self.num_nails = num_nails
        self.max_lines = max_lines
        self.img_size = img_size
        
        # 1. 读取与预处理
        img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图片: {target_img_path}")
            
        self.target_img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 误差矩阵: 255(白) -> 0, 0(黑) -> 255
        self.error_matrix = 255.0 - self.target_img.astype(float)
        
        # 初始化钉子
        self.nails = []
        center = (self.img_size / 2, self.img_size / 2)
        radius = self.img_size / 2 - 2
        for i in range(num_nails):
            angle = 2 * np.pi * i / num_nails
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            self.nails.append((np.clip(x, 0, self.img_size-1), np.clip(y, 0, self.img_size-1)))

        # 2. 预计算连线
        print(f"预计算连线中 (Size: {img_size}x{img_size})...")
        self.line_cache = {}
        for i in range(num_nails):
            for j in range(i + 1, num_nails):
                p1 = self.nails[i]
                p2 = self.nails[j]
                rr, cc, val = line_aa(p1[1], p1[0], p2[1], p2[0])
                valid = (rr >= 0) & (rr < self.img_size) & (cc >= 0) & (cc < self.img_size)
                self.line_cache[(i, j)] = (rr[valid], cc[valid], val[valid])
                self.line_cache[(j, i)] = (rr[valid], cc[valid], val[valid])
        
        self.path = [0] 
        self.line_weight = 10.0 
        
        # 用于存储每1000步的快照
        self.history_snapshots = []

    def run(self):
        print("开始绘图...")
        current_nail = self.path[0]
        final_canvas = np.zeros((self.img_size, self.img_size), dtype=float)

        for step in range(1, self.max_lines + 1):
            best_nail = -1
            max_score = -1.0
            
            # --- 贪心搜索 ---
            for next_nail in range(self.num_nails):
                if next_nail == current_nail: continue
                
                key = (current_nail, next_nail)
                if key not in self.line_cache: continue
                
                rr, cc, val = self.line_cache[key]
                
                # 算分
                pixels_error = self.error_matrix[rr, cc]
                score = np.sum(pixels_error * val)
                
                if score > max_score:
                    max_score = score
                    best_nail = next_nail

            # --- 应用选择 ---
            if best_nail != -1:
                self.path.append(best_nail)
                
                rr, cc, val = self.line_cache[(current_nail, best_nail)]
                
                self.error_matrix[rr, cc] -= self.line_weight * val
                self.error_matrix = np.clip(self.error_matrix, 0, 255)
                final_canvas[rr, cc] += self.line_weight * val
                
                current_nail = best_nail
                
                # --- 每1250步记录一次快照 ---
                if step % 1250 == 0:
                    print(f"--> 已完成 {step} 步，正在记录快照...")
                    # 必须使用 .copy()，否则所有历史记录都会指向同一个内存地址
                    snapshot = np.clip(final_canvas, 0, 255)
                    self.history_snapshots.append((step, snapshot.copy()))

            else:
                break
        
        print("计算完成! 正在生成展示图...")
        # 1. 展示过程大图
        self.show_evolution_grid()
        # 2. 展示最终结果
        self.show_final_compare(final_canvas)

    def show_evolution_grid(self):
        """将所有中间步骤汇合在一张大图里"""
        count = len(self.history_snapshots)
        if count == 0: return

        # 动态计算网格行列数 (例如 6张图 -> 2行3列)
        cols = 4 
        rows = math.ceil(count / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        fig.suptitle("Evolution Process (Every 1250 Steps)", fontsize=16)
        
        # 展平 axes 方便遍历，处理只有1行的情况
        if rows == 1 and cols == 1: axes = [axes]
        elif rows == 1 or cols == 1: axes = axes.flatten()
        else: axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < count:
                step_num, img_data = self.history_snapshots[i]
                # 反转颜色：白底黑线
                display_img = 255.0 - img_data
                ax.imshow(display_img, cmap='gray', vmin=0, vmax=255)
                ax.set_title(f"Step {step_num}")
                ax.axis('off')
            else:
                # 隐藏多余的空图表
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    def show_final_compare(self, canvas):
        """单独展示最终的高清对比"""
        canvas = np.clip(canvas, 0, 255)
        display_img = 255.0 - canvas
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(self.target_img, cmap='gray', vmin=0, vmax=255)
        plt.title("Original Target")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(display_img, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Final Result ({len(self.path)} lines)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        
        # 额外：保存最终结果到本地文件
        save_path = "string_art_result.png"
        cv2.imwrite(save_path, display_img)
        print(f"最终结果已保存为: {save_path}")

# --- 运行配置 ---
# 请修改为你的图片路径
target_image_path = r"C:\Users\86188\Desktop\python\20260128\1.png"

# max_lines 设置为 5000 或 6000，可以看到 5-6 张过程图
sa = EvolutionStringArt(target_image_path, num_nails=300, max_lines=10000, img_size=600)
sa.run()