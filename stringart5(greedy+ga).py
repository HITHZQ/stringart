import numpy as np
import cv2
import random
import time

class AdvancedStringArt:
    def __init__(self, target_img_path, n_pins=256, img_size=500, line_weight=20):
        self.n_pins = n_pins
        self.size = img_size
        self.line_weight = line_weight
        
        # 1. 预处理：加载图像
        raw_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raise ValueError(f"无法找到图像: {target_img_path}")
            
        # 调整大小并转为 float 类型防止溢出
        # 反转图像：我们希望背景是白(0)，线条是黑(255)，或者反过来
        # 这里逻辑：Canvas初始为0(黑)，Target处理为高亮代表线条密集处
        resized = cv2.resize(raw_img, (self.size, self.size)).astype(np.float32)
        self.target = 255.0 - resized # 反转：原图黑的地方现在数值高(255)，白的地方低(0)
        
        # 2. 初始化钉子
        self.pins = self._init_pins()
        
        # 3. 预计算查找表
        print(f"正在初始化... 图像大小: {self.size}x{self.size}, 钉子数: {self.n_pins}")
        t0 = time.time()
        self.line_cache = self._precompute_lines()
        print(f"预计算完成，耗时: {time.time()-t0:.2f}秒")

    def _init_pins(self):
        """生成圆周钉子坐标"""
        center = self.size // 2
        radius = self.size // 2 - 5
        pins = []
        for i in range(self.n_pins):
            angle = 2 * np.pi * i / self.n_pins
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            pins.append((x, y))
        return pins

    def _precompute_lines(self):
        """核心优化：预计算 Bresenham 直线路径索引"""
        cache = {}
        for i in range(self.n_pins):
            for j in range(i + 1, self.n_pins):
                mask = np.zeros((self.size, self.size), dtype=np.uint8)
                cv2.line(mask, self.pins[i], self.pins[j], 1, 1)
                rr, cc = np.where(mask > 0)
                cache[(i, j)] = cache[(j, i)] = (rr, cc)
        return cache

    def get_line_pixels(self, p1, p2):
        return self.line_cache.get((p1, p2), ([], []))

    def greedy_solve(self, max_lines=1500):
        """阶段1：贪婪算法快速构建轮廓"""
        print(f"--- 阶段 1: 贪婪算法 (Max Lines: {max_lines}) ---")
        canvas = np.zeros((self.size, self.size), dtype=np.float32)
        path = [0] # 从第0号钉子开始
        
        start_time = time.time()
        
        for l in range(max_lines):
            curr_pin = path[-1]
            best_pin = -1
            max_score = -float('inf')
            
            # 尝试所有可能的下一个钉子
            for next_pin in range(self.n_pins):
                if next_pin == curr_pin: continue
                # 跳过太近的钉子，避免边缘堆积
                if abs(next_pin - curr_pin) < 10 or abs(next_pin - curr_pin) > self.n_pins - 10:
                    continue
                
                rr, cc = self.get_line_pixels(curr_pin, next_pin)
                if len(rr) == 0: continue

                # 评分逻辑：这根线覆盖的区域，在 Target 中是否也是亮的？
                # 减去 Canvas 意味着：不要在这个地方重复画了
                # 简单理解：Sum(未被覆盖的目标亮度)
                val_target = np.sum(self.target[rr, cc])
                val_canvas = np.sum(canvas[rr, cc])
                
                # 如果这根线画上去，能填补多少未填补的“黑度”
                score = val_target - val_canvas
                
                if score > max_score:
                    max_score = score
                    best_pin = next_pin
            
            # 选中最佳路径
            path.append(best_pin)
            rr, cc = self.get_line_pixels(curr_pin, best_pin)
            canvas[rr, cc] += self.line_weight # 更新画布
            
            if l % 200 == 0:
                print(f"已生成 {l} 条线...")

        print(f"贪婪阶段完成，耗时: {time.time()-start_time:.2f}秒")
        return path, canvas

    def ga_refine(self, path, initial_canvas, iterations=3000):
        """阶段2：遗传算法 (模拟退火/爬山法) 精修"""
        
        print(f"--- 阶段 2: 遗传算法精修 (Iterations: {iterations}) ---")
        
        curr_path = list(path)
        curr_canvas = initial_canvas.copy()
        
        # 计算当前的全局误差 (MSE)
        curr_error = np.mean((self.target - curr_canvas) ** 2)
        print(f"初始误差 (MSE): {curr_error:.2f}")
        
        t0 = time.time()
        success_mutations = 0

        for i in range(iterations):
            # 1. 随机选择路径中间的一个节点进行突变
            idx = random.randint(1, len(curr_path) - 2)
            
            old_pin = curr_path[idx]
            prev_pin = curr_path[idx-1]
            next_pin = curr_path[idx+1]
            
            # 2. 随机选一个新的钉子
            new_pin = random.randint(0, self.n_pins - 1)
            if new_pin == old_pin: continue
            
            # 3. 增量更新：不重画全图，只操作受影响的两根线
            # 移除旧线: (prev -> old) 和 (old -> next)
            rr1, cc1 = self.get_line_pixels(prev_pin, old_pin)
            rr2, cc2 = self.get_line_pixels(old_pin, next_pin)
            
            # 临时画布
            temp_canvas = curr_canvas.copy()
            temp_canvas[rr1, cc1] -= self.line_weight
            temp_canvas[rr2, cc2] -= self.line_weight
            
            # 添加新线: (prev -> new) 和 (new -> next)
            rr3, cc3 = self.get_line_pixels(prev_pin, new_pin)
            rr4, cc4 = self.get_line_pixels(new_pin, next_pin)
            
            temp_canvas[rr3, cc3] += self.line_weight
            temp_canvas[rr4, cc4] += self.line_weight
            
            # 限制数值范围（防止负数或过曝），虽然Numpy会自动处理溢出，但clip更安全
            np.clip(temp_canvas, 0, 255, out=temp_canvas)
            
            # 4. 评估新误差
            # 优化：其实只需要计算受影响区域的误差变化，但为了代码清晰，这里算全图
            # 在高性能C++实现中，这里只计算局部 ROI
            new_error = np.mean((self.target - temp_canvas) ** 2)
            
            # 5. 选择机制：如果误差变小，就接受突变
            if new_error < curr_error:
                curr_path[idx] = new_pin
                curr_canvas = temp_canvas
                curr_error = new_error
                success_mutations += 1
            
            if i % 500 == 0:
                print(f"Iter {i}: 误差={curr_error:.2f}, 成功突变={success_mutations}")

        print(f"GA 完成，最终误差: {curr_error:.2f}")
        return curr_path, curr_canvas

    def save_result(self, canvas, filename="output.png"):
        # 将画布反转回白底黑线（符合人类审美）
        final_img = 255 - np.clip(canvas, 0, 255).astype(np.uint8)
        cv2.imwrite(filename, final_img)
        print(f"图片已保存至 {filename}")
        
        # 显示
        cv2.imshow("String Art", final_img)
        print("按任意键退出窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- 运行入口 ---
if __name__ == "__main__":
    # 请确保目录下有一张名为 '1.png' 的图片，或者修改这里
    # 如果没有图片，创建一个测试图片
    import os
    if not os.path.exists("1.png"):
        print("未找到 1.png，正在生成测试图片...")
        dummy = np.full((500, 500), 255, dtype=np.uint8)
        cv2.circle(dummy, (250, 250), 100, 0, -1) # 画个黑圆
        cv2.imwrite("1.png", dummy)

    solver = AdvancedStringArt("1.png", n_pins=300, img_size=600, line_weight=8)
    
    # 1. 贪婪跑一遍
    best_path, init_canvas = solver.greedy_solve(max_lines=10000)
    
    # 2. 遗传算法精修
    final_path, final_canvas = solver.ga_refine(best_path, init_canvas, iterations=500000)
    #收敛很慢，迭代次数可以根据需要调整，或者增加一些退火机制来加速收敛比如hill climbing
    # 3. 保存和显示
    solver.save_result(final_canvas)