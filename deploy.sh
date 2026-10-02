#!/bin/bash
# ============================================================
# AWS EC2 部署脚本（金奖版 v4）
# ============================================================
# 用法：把整个项目上传到 EC2 后，在项目目录运行
#   chmod +x deploy.sh && ./deploy.sh
#
# 功能：
# 1. 安装 Python 依赖
# 2. systemd 部署交易 Bot（开机自启、崩溃30秒自动重启）
# 3. systemd 部署监控 Dashboard
# ============================================================
set -e

echo "=========================================="
echo "APAC Quant Hackathon - 部署脚本 v4"
echo "=========================================="

# ---- 自动检测项目目录与运行用户 ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_USER="$(whoami)"
cd "$PROJECT_DIR"
echo "项目目录: $PROJECT_DIR"
echo "运行用户: $SERVICE_USER"

# 解析 python3 绝对路径
PYTHON_BIN="$(command -v python3)"
echo "Python: $PYTHON_BIN"

# ---- 1. 安装依赖 ----
echo "[1/4] 安装 Python 依赖..."
pip3 install -r requirements.txt -q

mkdir -p logs

# ---- 2. 交易 Bot systemd 服务 ----
echo "[2/4] 配置交易 Bot 服务..."
sudo tee /etc/systemd/system/quant-bot.service > /dev/null <<EOF
[Unit]
Description=APAC Quant Hackathon Trading Bot
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PYTHON_BIN $PROJECT_DIR/main.py
Restart=always
RestartSec=30
StandardOutput=append:$PROJECT_DIR/logs/bot_stdout.log
StandardError=append:$PROJECT_DIR/logs/bot_stderr.log

[Install]
WantedBy=multi-user.target
EOF

# ---- 3. Dashboard systemd 服务 ----
echo "[3/4] 配置 Dashboard 服务..."
sudo tee /etc/systemd/system/quant-dashboard.service > /dev/null <<EOF
[Unit]
Description=Quant Monitoring Dashboard
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PYTHON_BIN -m streamlit run $PROJECT_DIR/dashboard/app.py --server.headless true --server.port 8501
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload

# ---- 4. 启动 ----
echo "[4/4] 启动服务..."
sudo systemctl enable quant-bot quant-dashboard
sudo systemctl restart quant-bot
sudo systemctl restart quant-dashboard

echo ""
echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo "交易 Bot:  sudo systemctl status quant-bot"
echo "Dashboard: sudo systemctl status quant-dashboard"
echo "日志:      tail -f $PROJECT_DIR/logs/bot_stderr.log"
echo ""
echo "Dashboard 访问: http://<EC2公网IP>:8501"
echo "（需在 EC2 安全组放行 8501 端口）"
echo ""
echo "重启 Bot:  sudo systemctl restart quant-bot"
echo "停止 Bot:  sudo systemctl stop quant-bot"
