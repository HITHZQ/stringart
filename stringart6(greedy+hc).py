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

    def hill_climbing_refine(self, path, initial_canvas, iterations=3000):
        """
        优化版精修：爬山算法 (Hill Climbing)
        不再盲目随机，而是每次移除一个点后，寻找该位置的最佳替代点。
        """
        print(f"--- 阶段 2: 爬山算法精修 (Iterations: {iterations}) ---")
        
        curr_path = list(path)
        # 必须使用 float 类型以避免加减过程中的溢出/截断问题
        curr_canvas = initial_canvas.astype(np.float32)
        
        t0 = time.time()
        changes = 0

        # 为了防止死循环，我们创建一个随机访问的索引列表
        # 每次迭代处理路径上的一个节点
        check_indices = list(range(1, len(curr_path) - 1))
        
        for i in range(iterations):
            # 1. 随机选取路径中间的一个节点
            idx = random.choice(check_indices)
            
            old_pin = curr_path[idx]
            prev_pin = curr_path[idx-1]
            next_pin = curr_path[idx+1]
            
            # --- 关键步骤 A: 先“拔掉”旧钉子 ---
            # 从画布中减去旧的两根线：(prev -> old) 和 (old -> next)
            # 这会在画布上留出“空缺”，我们需要找到最好的新钉子来填补它
            rr1, cc1 = self.get_line_pixels(prev_pin, old_pin)
            rr2, cc2 = self.get_line_pixels(old_pin, next_pin)
            
            curr_canvas[rr1, cc1] -= self.line_weight
            curr_canvas[rr2, cc2] -= self.line_weight
            
            # --- 关键步骤 B: 寻找最佳替代钉子 ---
            best_pin = old_pin
            max_improvement = -float('inf')
            
            # 策略：不一定要遍历所有300个钉子，随机抽样 20-50 个通常足够
            # 甚至可以只搜索 old_pin 附近的邻居。
            # 这里为了演示极致效果，我们遍历所有钉子(N_PINS)，因为预计算过，速度很快。
            
            # 我们需要计算 "残差图" = Target - (Canvas - OldLines)
            # 我们希望新画的线落在残差图最亮（误差最大）的地方
            # 但直接计算全图太慢，我们只计算路径上的像素和
            
            candidates = range(self.n_pins) 
            # 如果觉得慢，可以取消下面这行的注释，改为只随机尝试 30 个点
            # candidates = random.sample(range(self.n_pins), 30)
            
            for candidate in candidates:
                if candidate == prev_pin or candidate == next_pin: continue
                
                # 获取两条新线的路径
                rr_a, cc_a = self.get_line_pixels(prev_pin, candidate)
                rr_b, cc_b = self.get_line_pixels(candidate, next_pin)
                
                if len(rr_a) == 0 or len(rr_b) == 0: continue

                # 极速评估：只看这两条线经过的区域
                # 评分 = (目标图该区域亮度) - (当前画布该区域已有亮度)
                # 这里的 curr_canvas 已经被减去了旧线，所以是“净画布”
                score_a = np.sum(self.target[rr_a, cc_a] - curr_canvas[rr_a, cc_a])
                score_b = np.sum(self.target[rr_b, cc_b] - curr_canvas[rr_b, cc_b])
                
                total_score = score_a + score_b
                
                if total_score > max_improvement:
                    max_improvement = total_score
                    best_pin = candidate

            # --- 关键步骤 C: 应用最佳选择 ---
            # 无论是否换了钉子，都要把选中的（原本的或新的）画回去
            curr_path[idx] = best_pin
            
            rr3, cc3 = self.get_line_pixels(prev_pin, best_pin)
            rr4, cc4 = self.get_line_pixels(best_pin, next_pin)
            
            curr_canvas[rr3, cc3] += self.line_weight
            curr_canvas[rr4, cc4] += self.line_weight
            
            if best_pin != old_pin:
                changes += 1

            if i % 500 == 0:
                # 这里的误差计算仅用于展示进度，不参与逻辑判断
                # 加上 clip 是为了计算准确的 MSE（避免负数影响）
                display_canvas = np.clip(curr_canvas, 0, 255)
                mse = np.mean((self.target - display_canvas) ** 2)
                print(f"Iter {i}: MSE={mse:.2f}, 优化次数={changes}")

        final_mse = np.mean((self.target - np.clip(curr_canvas, 0, 255)) ** 2)
        print(f"优化完成，最终 MSE: {final_mse:.2f}")
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
    
    # 2. 爬山算法精修
    final_path, final_canvas = solver.hill_climbing_refine(best_path, init_canvas, iterations=5000)
    
    # 3. 保存和显示
    solver.save_result(final_canvas)