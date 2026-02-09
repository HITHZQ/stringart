import numpy as np
import cv2
from skimage.draw import line_aa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
import time

class SliceVisualizer:
    def __init__(self, target_img_path, img_size=150):
        self.img_size = img_size
        img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError("无法读取图片")
        self.target_img = cv2.resize(img, (self.img_size, self.img_size))
        self.target_grid = 255.0 - self.target_img.astype(float)
        
    def run_benchmark_data(self, nails_list, lines_list, weight_list):
        """生成结构化数据，用于绘制网格"""
        
        # 结果字典，结构: data[weight][nails][lines] = (time, error)
        # 这种结构方便后续提取切片
        data_map = {}
        
        total = len(nails_list) * len(lines_list) * len(weight_list)
        count = 0
        
        print(f"开始计算数据，共 {total} 个组合...")
        
        # 为了后面画图方便，我们需要确保每一层的数据是完整的网格
        for w in weight_list:
            data_map[w] = {}
            for n in nails_list:
                data_map[w][n] = {}
                
                # 预计算钉子坐标 (针对当前的 n)
                nails = []
                center = (self.img_size/2, self.img_size/2)
                radius = self.img_size/2 - 2
                for i in range(n):
                    angle = 2 * np.pi * i / n
                    x = int(center[0] + radius * np.cos(angle))
                    y = int(center[1] + radius * np.sin(angle))
                    nails.append((np.clip(x, 0, self.img_size-1), np.clip(y, 0, self.img_size-1)))
                
                # 预计算连线缓存
                line_cache = {}
                for i in range(n):
                    for j in range(i + 1, n):
                        rr, cc, val = line_aa(nails[i][1], nails[i][0], nails[j][1], nails[j][0])
                        valid = (rr >= 0) & (rr < self.img_size) & (cc >= 0) & (cc < self.img_size)
                        line_cache[(i, j)] = (rr[valid], cc[valid], val[valid])
                        line_cache[(j, i)] = (rr[valid], cc[valid], val[valid])

                for l in lines_list:
                    count += 1
                    print(f"[{count}/{total}] 计算: W={w}, N={n}, L={l} ...", end="\r")
                    
                    # 运行绘图算法
                    t_start = time.time()
                    
                    error_matrix = self.target_grid.copy()
                    canvas = np.zeros((self.img_size, self.img_size))
                    curr_nail = 0
                    
                    # 加速步长
                    step = 2 if n > 200 else 1
                    
                    for _ in range(l):
                        best = -1
                        m_score = -1.0
                        for next_n in range(0, n, step):
                            if next_n == curr_nail: continue
                            key = (curr_nail, next_n)
                            if key not in line_cache: continue
                            rr, cc, val = line_cache[key]
                            score = np.sum(error_matrix[rr, cc] * val)
                            if score > m_score:
                                m_score = score
                                best = next_n
                        
                        if best != -1:
                            rr, cc, val = line_cache[(curr_nail, best)]
                            error_matrix[rr, cc] -= w * val
                            error_matrix = np.clip(error_matrix, 0, 255)
                            canvas[rr, cc] += w * val
                            curr_nail = best
                        else:
                            break
                            
                    duration = time.time() - t_start
                    
                    # 计算误差
                    gen = np.clip(canvas, 0, 255)
                    mae = np.mean(np.abs(self.target_img - (255-gen)))
                    
                    data_map[w][n][l] = (duration, mae)
                    
        print("\n计算完成，正在生成切片图...")
        self.plot_slices(nails_list, lines_list, weight_list, data_map)

    def plot_slices(self, nails_vals, lines_vals, weight_vals, data_map):
        fig = plt.figure(figsize=(18, 8))
        
        # 网格化 X, Y
        X_grid, Y_grid = np.meshgrid(nails_vals, lines_vals)
        
        # --- 子图1: 误差切片 (Error) ---
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_title("3D Slices: Image Error (Mean Absolute Error)")
        
        # 寻找全局最小最大值以统一颜色标尺
        all_errs = []
        for w in weight_vals:
            for n in nails_vals:
                for l in lines_vals:
                    all_errs.append(data_map[w][n][l][1])
        
        # 设置颜色映射 (紫=好, 黄=差)
        norm_err = colors.Normalize(vmin=min(all_errs), vmax=max(all_errs))
        cmap_err = cm.viridis 
        
        # 循环绘制每一层 Weight
        for w in weight_vals:
            # 构建这一层的高度 Z矩阵 (全都是 w)
            Z_layer = np.full_like(X_grid, w, dtype=float)
            
            # 构建这一层的颜色值 C矩阵
            C_layer = np.zeros_like(X_grid, dtype=float)
            
            # 填充数据 (注意 meshgrid 的索引顺序: Y对应行(lines), X对应列(nails))
            for i, l_val in enumerate(lines_vals):   # 行
                for j, n_val in enumerate(nails_vals): # 列
                    err = data_map[w][n_val][l_val][1]
                    C_layer[i, j] = err
            
            # 绘制表面
            # rstride, cstride 控制网格密度，越小越细
            surf = ax1.plot_surface(X_grid, Y_grid, Z_layer, 
                                    facecolors=cmap_err(norm_err(C_layer)),
                                    shade=False, alpha=0.85, rstride=1, cstride=1)

        ax1.set_xlabel('Nails (Count)')
        ax1.set_ylabel('Lines (Iterations)')
        ax1.set_zlabel('Weight (Thickness)')
        
        # 添加 Colorbar
        m_err = cm.ScalarMappable(cmap=cmap_err, norm=norm_err)
        m_err.set_array([])
        plt.colorbar(m_err, ax=ax1, shrink=0.5, aspect=10, label="Error (Lower is Better)")


        # --- 子图2: 时间切片 (Time) ---
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.set_title("3D Slices: Calculation Time (Seconds)")
        
        all_times = []
        for w in weight_vals:
            for n in nails_vals:
                for l in lines_vals:
                    all_times.append(data_map[w][n][l][0])
                    
        # 颜色映射 (紫=快, 黄=慢)
        norm_time = colors.Normalize(vmin=min(all_times), vmax=max(all_times))
        cmap_time = cm.plasma
        
        for w in weight_vals:
            Z_layer = np.full_like(X_grid, w, dtype=float)
            C_layer = np.zeros_like(X_grid, dtype=float)
            
            for i, l_val in enumerate(lines_vals):
                for j, n_val in enumerate(nails_vals):
                    t = data_map[w][n_val][l_val][0]
                    C_layer[i, j] = t
            
            ax2.plot_surface(X_grid, Y_grid, Z_layer, 
                             facecolors=cmap_time(norm_time(C_layer)),
                             shade=False, alpha=0.85, rstride=1, cstride=1)

        ax2.set_xlabel('Nails (Count)')
        ax2.set_ylabel('Lines (Iterations)')
        ax2.set_zlabel('Weight (Thickness)')

        m_time = cm.ScalarMappable(cmap=cmap_time, norm=norm_time)
        m_time.set_array([])
        plt.colorbar(m_time, ax=ax2, shrink=0.5, aspect=10, label="Time (Seconds)")

        # 调整视角以便看清层叠关系
        ax1.view_init(elev=20, azim=-60)
        ax2.view_init(elev=20, azim=-60)
        
        plt.tight_layout()
        plt.show()

# --- 运行配置 ---
if __name__ == "__main__":
    target_path = r"C:\Users\86188\Desktop\python\20260128\1.png"
    
    # 定义轴数据
    # 为了让"片状"效果明显，X和Y需要一定的密度，而Z(Weight)需要间隔开
    
    nails = [100, 150, 200, 250, 300, 350]       # X轴
    lines = [1000, 2000, 3000, 4000, 5000, 6000]   # Y轴
    weights = [2.5, 5, 7.5, 10, 12.5, 15]            # Z轴 (层数)
    

    vis = SliceVisualizer(target_path, img_size=150)
    vis.run_benchmark_data(nails, lines, weights)