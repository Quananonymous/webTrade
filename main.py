# main.py
from trading_bot_lib import BotManager, STATE_MANAGER, RSIEMABot, EMACrossoverBot, Reverse24hBot, TrendFollowingBot, ScalpingBot, SafeGridBot
import os
import json
import time
import threading
from flask import Flask, jsonify, request
import logging

# Thiết lập logging cho luồng chính
logger = logging.getLogger(__name__)

# =========================================================
# CẤU HÌNH VÀ BIẾN MÔI TRƯỜNG
# =========================================================

# Railway sử dụng biến môi trường PORT. Mặc định là 5000 cho local.
PORT = int(os.getenv('PORT', 5000))
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
bot_config_json = os.getenv('BOT_CONFIGS', '[]')

try:
    BOT_CONFIGS = json.loads(bot_config_json)
except Exception as e:
    logger.error(f"Lỗi phân tích cấu hình BOT_CONFIGS: {e}")
    BOT_CONFIGS = []

# =========================================================
# KHỞI TẠO GLOBAL MANAGER VÀ FLASK
# =========================================================

manager: BotManager = None 
app = Flask(__name__)

# MAPPER TỪ TÊN CHIẾN LƯỢC SANG LỚP BOT
STRATEGY_MAPPER = {
    "RSI/EMA Recursive": RSIEMABot,
    "EMA Crossover": EMACrossoverBot,
    "Reverse 24h": Reverse24hBot,
    "Trend Following": TrendFollowingBot,
    "Scalping": ScalpingBot,
    "Safe Grid": SafeGridBot
}

def init_manager(api_key, api_secret):
    """Khởi tạo hoặc khởi động lại BotManager và thêm cấu hình mặc định."""
    global manager
    
    if manager:
        manager.stop_all() # Dừng hệ thống cũ
        
    manager = BotManager(
        api_key=api_key,
        api_secret=api_secret,
        state_manager=STATE_MANAGER
    )
    
    # Thêm bot từ cấu hình mặc định
    if BOT_CONFIGS:
        logger.info(f"Đang thêm {len(BOT_CONFIGS)} bot từ cấu hình mặc định...")
        for config in BOT_CONFIGS:
            symbol = config.get('symbol', None)
            strategy = config.get('strategy')
            lev = config.get('lev')
            percent = config.get('percent')
            tp = config.get('tp')
            sl = config.get('sl', 0)
            threshold = config.get('threshold', 30)

            if strategy and lev and percent and tp:
                manager.add_bot(symbol, lev, percent, tp, sl, strategy, threshold=threshold)
            else:
                logger.warning(f"Cấu hình bot bị thiếu tham số. Bỏ qua: {config}")
    
    return True

# =========================================================
# API ENDPOINTS
# =========================================================

@app.route('/api/init', methods=['POST'])
def init_bot_manager_api():
    """Endpoint để khởi tạo/cập nhật API Key và khởi động hệ thống."""
    data = request.json
    # Ưu tiên API Key từ request body, sau đó là biến môi trường
    api_key = data.get('api_key', BINANCE_API_KEY)
    api_secret = data.get('api_secret', BINANCE_SECRET_KEY)
    
    if not api_key or not api_secret:
        return jsonify({"success": False, "message": "API Key hoặc Secret Key bị thiếu"}), 400
        
    try:
        if init_manager(api_key, api_secret):
            return jsonify({"success": True, "message": "Hệ thống bot đã được khởi tạo/cập nhật API Key thành công và bot mặc định đã chạy."}), 200
        else:
            return jsonify({"success": False, "message": "Lỗi khởi tạo không xác định."}), 500
    except Exception as e:
        logger.error(f"Lỗi khởi tạo hệ thống: {str(e)}")
        return jsonify({"success": False, "message": f"Lỗi khởi tạo bot: {str(e)}"}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Endpoint trả về trạng thái của tất cả các bot và hệ thống."""
    if not manager:
        return jsonify({"system_status": "Inactive", "message": "Hệ thống chưa được khởi tạo. Vui lòng gửi API Key đến /api/init"}), 200
        
    # Lấy số dư và trạng thái kết nối
    balance = manager._verify_api_connection()
    
    return jsonify({
        "system_status": "Running" if manager.running else "Stopped",
        "binance_connection": "OK" if balance is not None else "Error",
        "binance_balance_usdt": f"{balance:.2f}" if balance is not None else "N/A",
        "active_bots_count": len(STATE_MANAGER.bot_statuses),
        "bot_statuses": STATE_MANAGER.bot_statuses
    })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Endpoint trả về lịch sử logs/giao dịch."""
    return jsonify(STATE_MANAGER.trade_logs)

@app.route('/api/add', methods=['POST'])
def add_bot_web():
    """Endpoint để thêm một bot mới."""
    if not manager:
        return jsonify({"success": False, "message": "Hệ thống chưa khởi tạo. Vui lòng gửi API Key đến /api/init"}), 400
        
    config = request.json
    symbol = config.get('symbol', None)
    strategy = config.get('strategy')
    lev = config.get('lev')
    percent = config.get('percent')
    tp = config.get('tp')
    sl = config.get('sl', 0)
    threshold = config.get('threshold', 30)

    if not all([strategy, lev, percent, tp]):
        return jsonify({"success": False, "message": "Thiếu các tham số bắt buộc (strategy, lev, percent, tp)"}), 400

    if manager.add_bot(symbol, lev, percent, tp, sl, strategy, threshold=threshold):
        return jsonify({"success": True, "message": f"Đã thêm bot {strategy} cho {symbol if symbol else 'AUTO'} thành công."}), 200
    else:
        return jsonify({"success": False, "message": f"Không thể thêm bot {strategy}. Kiểm tra log hệ thống."}), 500

@app.route('/api/stop/<string:bot_id>', methods=['POST'])
def stop_bot_web(bot_id):
    """Endpoint để dừng một bot theo ID."""
    if not manager:
        return jsonify({"success": False, "message": "Hệ thống chưa khởi tạo."}), 400

    if manager.stop_bot(bot_id):
        return jsonify({"success": True, "message": f"Đã dừng bot {bot_id} thành công."}), 200
    else:
        return jsonify({"success": False, "message": f"Không tìm thấy bot {bot_id} đang chạy."}), 404

# =========================================================
# KHỞI ĐỘNG SERVER
# =========================================================

def run_flask_server():
    """Hàm chạy Flask server."""
    
    # Nếu có API Key trong biến môi trường Railway, khởi tạo ngay
    if BINANCE_API_KEY and BINANCE_SECRET_KEY:
        try:
            init_manager(BINANCE_API_KEY, BINANCE_SECRET_KEY)
            logger.info("BotManager khởi tạo tự động từ biến môi trường thành công.")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo BotManager tự động: {e}")
            
    logger.info(f"🌐 Đang khởi động Web Server tại 0.0.0.0:{PORT}")
    # Chạy Flask Server với PORT của Railway
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False) 

if __name__ == "__main__":
    run_flask_server()
