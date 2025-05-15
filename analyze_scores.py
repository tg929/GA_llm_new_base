import numpy as np

# 读取文件内容
with open('output_dockutils_runga_test0/5ht1b/generation_0_docking/docked_5ht1b.smi', 'r') as f:
    lines = f.readlines()

# 提取得分
scores = []
for line in lines:
    parts = line.strip().split('\t')
    if len(parts) == 2:
        try:
            score = float(parts[1])
            scores.append(score)
        except ValueError:
            continue

# 转换为numpy数组并排序
scores = np.array(scores)
scores.sort()

# 计算统计信息
total_mean = np.mean(scores)
top100_mean = np.mean(scores[:100])
top50_mean = np.mean(scores[:50])
top20_mean = np.mean(scores[:20])
top10_mean = np.mean(scores[:10])
top1_mean = scores[0]

print(f"总分子数: {len(scores)}")
print(f"所有分子平均得分: {total_mean:.2f}")
print(f"Top 100平均得分: {top100_mean:.2f}")
print(f"Top 50平均得分: {top50_mean:.2f}")
print(f"Top 20平均得分: {top20_mean:.2f}")
print(f"Top 10平均得分: {top10_mean:.2f}")
print(f"Top 1得分: {top1_mean:.2f}") 