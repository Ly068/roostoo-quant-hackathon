#!/bin/bash
# ============================================================
# 实时监控：每10秒刷新 组合市值/持仓/仓位/最新日志
# 在 Session Manager 终端运行：  bash monitor.sh
# 按 Ctrl+C 退出监控（Bot 在 tmux 后台继续运行，不受影响）
# ============================================================
cd ~/roostoo-quant-hackathon || exit 1
while true; do
  clear
  echo "================ 交易 Bot 实时监控  (UTC $(date -u '+%Y-%m-%d %H:%M:%S')) ================"
  python3 scripts/status_snapshot.py
  echo "---------------------------- 最近 6 条日志 ----------------------------"
  tail -n 6 data/live/console.log
  echo "======================================================================="
  echo "  每 10 秒自动刷新 | 按 Ctrl+C 退出监控（Bot 继续运行）"
  sleep 10
done
