#!/bin/bash
# ============================================================
# AWS 一键环境准备脚本（Amazon Linux 2023 / Session Manager）
# 在 Session Manager 终端运行：  bash setup_aws.sh
# 作用：装 tmux -> 拉代码 -> 装依赖 -> 生成配置模板
# ============================================================
set -e
REPO="https://github.com/Ly068/roostoo-quant-hackathon.git"
DIR="roostoo-quant-hackathon"

echo ">>> [1/5] 系统更新并安装 tmux / pip"
sudo dnf update -y
sudo dnf install -y tmux python3-pip || true

echo ">>> [2/5] 获取最新代码"
cd ~
if [ -d "$DIR/.git" ]; then
  cd "$DIR" && git pull
else
  git clone "$REPO" && cd "$DIR"
fi

echo ">>> [3/5] 安装 Python 依赖"
pip3 install -r requirements.txt --user 2>/dev/null || pip3 install -r requirements.txt

echo ">>> [4/5] 准备配置文件"
if [ ! -f config.yaml ]; then
  cp config.example.yaml config.yaml
  echo "    已从模板生成 config.yaml（待填入密钥）"
else
  echo "    config.yaml 已存在，保留不动"
fi

echo ">>> [5/5] 环境准备完成"
echo "============================================================"
echo "下一步（务必按顺序）："
echo ""
echo "  1) 编辑配置，填入 API 密钥；开赛当天把 USE_TESTNET 改为 false"
echo "       nano ~/$DIR/config.yaml"
echo ""
echo "  2) 一键在 tmux 后台启动 Bot"
echo "       bash start_bot.sh"
echo ""
echo "  3) 查看运行 / 停止"
echo "       tmux attach -t quant     # 查看（Ctrl+B 再 D 返回）"
echo "       tmux kill-session -t quant"
echo "============================================================"
