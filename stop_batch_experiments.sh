# 脚本：stop_batch_experiments.sh

#!/bin/bash

OUTPUT_DIR_PATH="/data1/ytg/GA_llm_new/output_batch_runs" # 请替换成你的实际路径

echo "监控并终止向目录 $OUTPUT_DIR_PATH 写入的进程..."
echo "按 Ctrl+C 停止监控。"

while true; do
  # 查找所有在指定目录下打开了文件的进程PID
  # 注意：这里假设进程会直接在 batch_experiments 目录下创建或写入文件。
  # 如果进程是将文件写入子目录，lsof 的参数可能需要调整，或者使用更复杂的查找。
  PIDS_TO_KILL=$(lsof +D "$OUTPUT_DIR_PATH" 2>/dev/null | awk 'NR>1 {print $2}' | sort -u)

  if [ -n "$PIDS_TO_KILL" ]; then
    echo "$(date): 发现以下进程正在使用目录 $OUTPUT_DIR_PATH:"
    echo "$PIDS_TO_KILL"
    echo "正在终止这些进程..."
    # 使用 kill -9 强制终止，请谨慎使用，它不会给进程清理的机会
    # 可以先尝试 kill (SIGTERM)，如果无效再用 kill -9 (SIGKILL)
    echo "$PIDS_TO_KILL" | xargs kill -9
    echo "进程已终止。"
  else
    echo -n "." # 输出一个点表示正在监控，避免屏幕空白
  fi
  sleep 5 # 每5秒检查一次，你可以根据需要调整这个间隔
done