import numpy as np

#自适应分裂速度控制
# 最终迭代轮次iteration_end
# 开始迭代轮次iteration_start
# 每densification_interval分裂一次
# 当前迭代轮次iteration
# 初始点数initial_num_points
# 当前点数current_num_points
# 最终要达到点数final_num_points
# 根据当前迭代轮次，计算过去分裂了多少次，还有增加的点数，根据这个点数来计算分裂速度
# 用剩余的迭代轮次，计算还会分裂多少次
# 根据历史速度，调整分裂速度
# 分裂速度与点数正相关的一个数字
class AdaptiveSplitting:
    def set_params(self, final_num_points, exclude_num_points, iteration_end, densification_interval,  split_method="log"):
        #要排除的点数，比如天空球，包围盒外等等
        self.exclude_num_points = exclude_num_points
        self.final_num_points = final_num_points - exclude_num_points
        self.iteration_end = iteration_end
        self.densification_interval = densification_interval
       
        self.initial_speed = 0.01
        self.split_method = split_method

    def calculate_split_speed_exponential(self):
        #print("calculate_split_speed_exponential")
        # 根据history_nums计算历史分裂速度
        # 点的增加，基本符合对数增长，所以用对数来计算
        # 根据，初始点数，当前利率（速度），剩余的迭代轮次，计算最终点数
        #  num_points = self.current_num_points * (1+speed) ** remaining_iterations
        remaining_iterations = np.ceil((self.iteration_end - self.current_iteration) / self.densification_interval)

        growth_factor = self.final_num_points / self.current_num_points
        speed = np.exp(np.log(growth_factor) / remaining_iterations) - 1
        if not (0 <= speed <= 1):
            speed = self.initial_speed
        return speed

    def calculate_split_speed_linear(self):
        #print("calculate_split_speed_linear")
        # 剩余的迭代轮次
        remaining_iterations = np.ceil((self.iteration_end - self.current_iteration) / self.densification_interval)
        # 剩余需要增加的点数
        remaining_points = self.final_num_points - self.current_num_points

        # 线性增长速度
        if remaining_iterations > 0 and self.current_num_points > 0:
            speed = remaining_points / (remaining_iterations * self.current_num_points)
        else:
            speed = 0  # 如果已经达到目标点数或迭代结束，速度为0

        # 限制速度范围
        speed = np.clip(speed, 0, 1)
        return speed
    def calculate_split_speed_logarithmic(self):
        """
        对数型分裂速度（示例思路）：
        希望剩余迭代次数为 remaining_iterations 时，能从 current_num_points 
        “对数式”增长到 final_num_points。这里提供一种简单的思路：假设
            final_num_points = current_num_points * [1 + speed * ln(remaining_iterations + 1)]
        通过该公式解出 speed：
            speed = (final_num_points / current_num_points - 1) / ln(remaining_iterations + 1)
        并限制到 [0, 1] 范围内。如果需要更复杂的 log 型分裂曲线，可自行修改。
        """
        remaining_iterations = np.ceil((self.iteration_end - self.current_iteration) / self.densification_interval)
        if remaining_iterations <= 0:
            return 0.0

        growth_factor = self.final_num_points / max(self.current_num_points, 1e-8)
        if growth_factor <= 1:
            # 当前点数已满足或超过目标，则无需再增长
            return 0.0

        # 计算速度
        # 避免 ln(1) = 0 的问题，因此加上 +1。
        denom = np.log(remaining_iterations + 1)
        # 若 denom 为 0 或接近 0，会引发数值问题，这里做个保护
        if abs(denom) < 1e-8:
            return 0.0

        speed = (growth_factor - 1) / denom
        # 将速度限制在 [0, 1] 之间
        speed = np.clip(speed, 0, 1)
        return speed
    def update(self, current_num_points, current_iteration):
        self.current_iteration = current_iteration
        self.current_num_points = min(current_num_points - self.exclude_num_points, self.final_num_points)
        assert self.current_num_points >= 0, "current_num_points should not be negative"

        if self.split_method == 'linear':
            split_speed = self.calculate_split_speed_linear()
        elif self.split_method == 'exponential':
            split_speed = self.calculate_split_speed_exponential()
        elif self.split_method == 'log':
            split_speed = self.calculate_split_speed_logarithmic()
        elif self.split_method == 'auto':
            if self.current_iteration < 0.7 * self.iteration_end:
                split_speed = self.calculate_split_speed_exponential()
            else:
                split_speed = self.calculate_split_speed_linear()
        else:
            raise ValueError(f"Invalid split method: {self.split_method}")


        return split_speed
           
