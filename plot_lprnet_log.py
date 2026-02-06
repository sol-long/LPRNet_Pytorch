import re
import matplotlib.pyplot as plt

# ================= 配置 =================
LOG_FILE = "train_log.txt"  # 你的日志文件名
SMOOTH_FACTOR = 0.9         # 平滑系数 (0~1)，让曲线更圆滑好读
# =======================================

def smooth(scalars, weight):
    """类似 TensorBoard 的平滑算法"""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    losses = []
    iters = []
    accuracies = []
    acc_epochs = []

    # 1. 读取日志文件
    try:
        with open(LOG_FILE, "r", encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ 找不到文件: {LOG_FILE}")
        return

    # 2. 正则解析
    # 匹配: Epoch:1 || epochiter: 50/100 || ... Loss: 0.1234 ...
    loss_pattern = re.compile(r"Loss:\s*([\d\.]+)")
    
    # 匹配: [Info] Test Accuracy: 0.95 ...
    acc_pattern = re.compile(r"Test Accuracy:\s*([\d\.]+)")

    iter_count = 0
    current_epoch = 0
    
    for line in lines:
        # 提取 Loss
        if "Loss:" in line:
            match = loss_pattern.search(line)
            if match:
                losses.append(float(match.group(1)))
                iter_count += 1
                iters.append(iter_count)
        
        # 提取 Accuracy (通常每个Epoch测一次)
        if "Test Accuracy:" in line:
            match = acc_pattern.search(line)
            if match:
                accuracies.append(float(match.group(1)))
                current_epoch += 1
                acc_epochs.append(current_epoch)

    if not losses:
        print("⚠️ 未找到 Loss 数据，请检查日志格式是否包含 'Loss: x.xxxx'")
        return

    # 3. 绘图
    plt.figure(figsize=(12, 5))

    # --- 左图：CTC Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(iters, losses, alpha=0.3, color='orange', label='Raw')
    if len(losses) > 10:
        plt.plot(iters, smooth(losses, SMOOTH_FACTOR), color='red', label='Smooth')
    plt.title("Training Loss (CTC)")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- 右图：Test Accuracy ---
    plt.subplot(1, 2, 2)
    if accuracies:
        plt.plot(acc_epochs, accuracies, 'b-o', linewidth=2, label='Accuracy')
        plt.title("Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (0~1)")
        plt.ylim(0, 1.05) # 限制 y 轴范围
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 标出最高点
        max_acc = max(accuracies)
        max_idx = accuracies.index(max_acc)
        plt.text(acc_epochs[max_idx], max_acc, f" Max: {max_acc:.4f}", fontsize=10, verticalalignment='bottom')
    else:
        plt.text(0.5, 0.5, "No Accuracy Data Found", ha='center')

    plt.tight_layout()
    plt.savefig("LPRNet_Results.png", dpi=150)
    print("✅ 绘图完成！已保存为 LPRNet_Results.png")
    plt.show()

if __name__ == "__main__":
    main()