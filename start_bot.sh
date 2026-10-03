#!/bin/bash
# ============================================================
# 在 tmux 后台会话 "quant" 中启动交易 Bot（断开终端也持续运行）
# 在 Session Manager 终端运行：  bash start_bot.sh
# ============================================================
DIR="roostoo-quant-hackathon"
cd ~/"$DIR"
mkdir -p data/live

# 若已存在同名会话，先关闭
tmux kill-session -t quant 2>/dev/null || true

# 检查 config.yaml 是否已填密钥
if grep -qE '^TEST_API_KEY: ""' config.yaml; then
  echo "!! config.yaml 还没填入 API 密钥，请先运行：  nano config.yaml"
  exit 1
fi

tmux new-session -d -s quant \
  "python3 main.py 2>&1 | tee data/live/console.log; exec bash"

sleep 3
echo "✓ Bot 已在 tmux 会话 [quant] 后台启动"
tmux ls
echo ""
echo "  实时查看： tmux attach -t quant   （Ctrl+B 再按 D 返回）"
echo "  查看日志： tail -f data/live/console.log"
echo "  停止 Bot： tmux kill-session -t quant"
