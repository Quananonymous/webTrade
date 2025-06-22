import json
import hmac
import hashlib
import time
import threading
import urllib.request
import urllib.parse
import numpy as np
import websocket
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Cấu hình logging chi tiết
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot_errors.log')
    ]
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Tạo ứng dụng Flask
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Lấy cấu hình từ biến môi trường
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    
# Cấu hình bot từ biến môi trường (dạng JSON)
bot_config_json = os.getenv('BOT_CONFIGS', '[]')
try:
    BOT_CONFIGS = json.loads(bot_config_json)
except Exception as e:
    logging.error(f"Lỗi phân tích cấu hình BOT_CONFIGS: {e}")
    BOT_CONFIGS = []

API_KEY = BINANCE_API_KEY
API_SECRET = BINANCE_SECRET_KEY

# ========== HÀM HỖ TRỢ API BINANCE VỚI XỬ LÝ LỖI CHI TIẾT ==========
def sign(query):
    try:
        return hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    except Exception as e:
        logger.error(f"Lỗi tạo chữ ký: {str(e)}")
        return ""

def binance_api_request(url, method='GET', params=None, headers=None):
    """Hàm tổng quát cho các yêu cầu API Binance với xử lý lỗi chi tiết"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if method.upper() == 'GET':
                if params:
                    query = urllib.parse.urlencode(params)
                    url = f"{url}?{query}"
                req = urllib.request.Request(url, headers=headers or {})
            else:
                data = urllib.parse.urlencode(params).encode() if params else None
                req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
            
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 200:
                    return json.loads(response.read().decode())
                else:
                    logger.error(f"Lỗi API ({response.status}): {response.read().decode()}")
                    if response.status == 429:  # Rate limit
                        time.sleep(2 ** attempt)  # Exponential backoff
                    elif response.status >= 500:
                        time.sleep(1)
                    continue
        except urllib.error.HTTPError as e:
            logger.error(f"Lỗi HTTP ({e.code}): {e.reason}")
            if e.code == 429:  # Rate limit
                time.sleep(2 ** attempt)  # Exponential backoff
            elif e.code >= 500:
                time.sleep(1)
            continue
        except Exception as e:
            logger.error(f"Lỗi kết nối API: {str(e)}")
            time.sleep(1)
    
    logger.error(f"Không thể thực hiện yêu cầu API sau {max_retries} lần thử")
    return None

def get_step_size(symbol):
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        data = binance_api_request(url)
        if not data:
            return 0.001
            
        for s in data['symbols']:
            if s['symbol'] == symbol.upper():
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        return float(f['stepSize'])
    except Exception as e:
        logger.error(f"Lỗi lấy step size: {str(e)}")
    return 0.001

def set_leverage(symbol, lev):
    try:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "leverage": lev,
            "timestamp": ts
        }
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/leverage?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        response = binance_api_request(url, method='POST', headers=headers)
        if response and 'leverage' in response:
            return True
    except Exception as e:
        logger.error(f"Lỗi thiết lập đòn bẩy: {str(e)}")
    return False

def get_balance():
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        data = binance_api_request(url, headers=headers)
        if not data:
            return 0
            
        for asset in data['assets']:
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])
    except Exception as e:
        logger.error(f"Lỗi lấy số dư: {str(e)}")
    return 0

def place_order(symbol, side, qty):
    try:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "side": side,
            "type": "MARKET",
            "quantity": qty,
            "timestamp": ts
        }
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/order?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        return binance_api_request(url, method='POST', headers=headers)
    except Exception as e:
        logger.error(f"Lỗi đặt lệnh: {str(e)}")
    return None

def cancel_all_orders(symbol):
    try:
        ts = int(time.time() * 1000)
        params = {"symbol": symbol.upper(), "timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/allOpenOrders?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        binance_api_request(url, method='DELETE', headers=headers)
        return True
    except Exception as e:
        logger.error(f"Lỗi hủy lệnh: {str(e)}")
    return False

def get_current_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data and 'price' in data:
            return float(data['price'])
    except Exception as e:
        logger.error(f"Lỗi lấy giá: {str(e)}")
    return 0

def get_positions(symbol=None):
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        if symbol:
            params["symbol"] = symbol.upper()
            
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v2/positionRisk?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        positions = binance_api_request(url, headers=headers)
        if not positions:
            return []
            
        if symbol:
            for pos in positions:
                if pos['symbol'] == symbol.upper():
                    return [pos]
            
        return positions
    except Exception as e:
        logger.error(f"Lỗi lấy vị thế: {str(e)}")
    return []

# ========== TÍNH CHỈ BÁO KỸ THUẬT VỚI KIỂM TRA DỮ LIỆU ==========
def calc_rsi(prices, period=14):
    try:
        if len(prices) < period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1 + rs))
    except Exception as e:
        logger.error(f"Lỗi tính RSI: {str(e)}")
        return None

# ========== QUẢN LÝ WEBSOCKET HIỆU QUẢ VỚI KIỂM SOÁT LỖI ==========
class WebSocketManager:
    def __init__(self):
        self.connections = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
    def add_symbol(self, symbol, callback):
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self.connections:
                self._create_connection(symbol, callback)
                
    def _create_connection(self, symbol, callback):
        if self._stop_event.is_set():
            return
            
        stream = f"{symbol.lower()}@trade"
        url = f"wss://fstream.binance.com/ws/{stream}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'p' in data:
                    price = float(data['p'])
                    self.executor.submit(callback, price)
            except Exception as e:
                logger.error(f"Lỗi xử lý tin nhắn WebSocket {symbol}: {str(e)}")
                
        def on_error(ws, error):
            logger.error(f"Lỗi WebSocket {symbol}: {str(error)}")
            if not self._stop_event.is_set():
                time.sleep(5)
                self._reconnect(symbol, callback)
            
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket đóng {symbol}: {close_status_code} - {close_msg}")
            if not self._stop_event.is_set() and symbol in self.connections:
                time.sleep(5)
                self._reconnect(symbol, callback)
                
        ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        
        self.connections[symbol] = {
            'ws': ws,
            'thread': thread,
            'callback': callback
        }
        logger.info(f"WebSocket bắt đầu cho {symbol}")
        
    def _reconnect(self, symbol, callback):
        logger.info(f"Kết nối lại WebSocket cho {symbol}")
        self.remove_symbol(symbol)
        self._create_connection(symbol, callback)
        
    def remove_symbol(self, symbol):
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.connections:
                try:
                    self.connections[symbol]['ws'].close()
                except Exception as e:
                    logger.error(f"Lỗi đóng WebSocket {symbol}: {str(e)}")
                del self.connections[symbol]
                logger.info(f"WebSocket đã xóa cho {symbol}")
                
    def stop(self):
        self._stop_event.set()
        for symbol in list(self.connections.keys()):
            self.remove_symbol(symbol)

# ========== BOT CHÍNH VỚI ĐÓNG LỆNH CHÍNH XÁC ==========
class IndicatorBot:
    def __init__(self, symbol, lev, percent, tp, sl, indicator, ws_manager):
        self.symbol = symbol.upper()
        self.lev = lev
        self.percent = percent
        self.tp = tp
        self.sl = sl
        self.indicator = indicator
        self.ws_manager = ws_manager
        self.status = "waiting"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self.prices = []
        self._stop = False
        self.position_open = False
        self.last_trade_time = 0
        self.last_rsi = 50
        self.position_check_interval = 60
        self.last_position_check = 0
        self.last_error_log_time = 0
        self.last_close_time = 0
        self.cooldown_period = 60
        self.max_position_attempts = 3
        self.position_attempt_count = 0
        
        # Đăng ký với WebSocket Manager
        self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        
        # Bắt đầu thread chính
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"🟢 Bot khởi động cho {self.symbol}")

    def log(self, message):
        """Ghi log hệ thống"""
        logger.info(f"[{self.symbol}] {message}")

    def _handle_price_update(self, price):
        if self._stop: 
            return
            
        self.prices.append(price)
        if len(self.prices) > 100:
            self.prices = self.prices[-100:]

    def _run(self):
        """Luồng chính quản lý bot"""
        while not self._stop:
            try:
                current_time = time.time()
                
                if current_time - self.last_position_check > self.position_check_interval:
                    self.check_position_status()
                    self.last_position_check = current_time
                
                if not self.position_open and self.status == "waiting":
                    if current_time - self.last_close_time < self.cooldown_period:
                        time.sleep(1)
                        continue
                    
                    signal = self.get_signal()
                    
                    if signal and current_time - self.last_trade_time > 60:
                        self.open_position(signal)
                        self.last_trade_time = current_time
                
                if self.position_open and self.status == "open":
                    self.check_tp_sl()
                
                time.sleep(1)
                
            except Exception as e:
                if time.time() - self.last_error_log_time > 10:
                    self.log(f"Lỗi hệ thống: {str(e)}")
                    self.last_error_log_time = time.time()
                time.sleep(5)

    def stop(self):
        self._stop = True
        self.ws_manager.remove_symbol(self.symbol)
        try:
            cancel_all_orders(self.symbol)
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"Lỗi hủy lệnh: {str(e)}")
                self.last_error_log_time = time.time()
        logger.info(f"🔴 Bot dừng cho {self.symbol}")

    def check_position_status(self):
        """Kiểm tra trạng thái vị thế"""
        try:
            positions = get_positions(self.symbol)
            
            if not positions or len(positions) == 0:
                self.position_open = False
                self.status = "waiting"
                self.side = ""
                self.qty = 0
                self.entry = 0
                return
            
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    position_amt = float(pos.get('positionAmt', 0))
                    
                    if abs(position_amt) > 0:
                        self.position_open = True
                        self.status = "open"
                        self.side = "BUY" if position_amt > 0 else "SELL"
                        self.qty = position_amt
                        self.entry = float(pos.get('entryPrice', 0))
                        return
            
            self.position_open = False
            self.status = "waiting"
            self.side = ""
            self.qty = 0
            self.entry = 0
            
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"Lỗi kiểm tra vị thế: {str(e)}")
                self.last_error_log_time = time.time()

    def check_tp_sl(self):
        """Tự động kiểm tra và đóng lệnh khi đạt TP/SL"""
        if not self.position_open or not self.entry or not self.qty:
            return
            
        try:
            if len(self.prices) > 0:
                current_price = self.prices[-1]
            else:
                current_price = get_current_price(self.symbol)
                
            if current_price <= 0:
                return
                
            if self.side == "BUY":
                profit = (current_price - self.entry) * self.qty
            else:
                profit = (self.entry - current_price) * abs(self.qty)
                
            invested = self.entry * abs(self.qty) / self.lev
            if invested <= 0:
                return
                
            roi = (profit / invested) * 100
            
            if roi >= self.tp:
                self.close_position(f"✅ Đạt TP {self.tp}% (ROI: {roi:.2f}%)")
            elif roi <= -self.sl:
                self.close_position(f"❌ Đạt SL {self.sl}% (ROI: {roi:.2f}%)")
                
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"Lỗi kiểm tra TP/SL: {str(e)}")
                self.last_error_log_time = time.time()

    def get_signal(self):
        if len(self.prices) < 40:
            return None
            
        prices_arr = np.array(self.prices)
        rsi_val = calc_rsi(prices_arr)
        
        if rsi_val is not None:
            self.last_rsi = rsi_val
            if rsi_val <= 30: 
                return "BUY"
            if rsi_val >= 70: 
                return "SELL"
                    
        return None

    def open_position(self, side):
        self.check_position_status()
        
        if self.position_open:
            self.log(f"⚠️ Đã có vị thế mở, không vào lệnh mới")
            return
            
        try:
            cancel_all_orders(self.symbol)
            
            if not set_leverage(self.symbol, self.lev):
                self.log(f"Không thể đặt đòn bẩy {self.lev}")
                return
            
            balance = get_balance()
            if balance <= 0:
                self.log(f"Không đủ số dư USDT")
                return
            
            if self.percent > 100:
                self.percent = 100
            elif self.percent < 1:
                self.percent = 1
                
            usdt_amount = balance * (self.percent / 100)
            price = get_current_price(self.symbol)
            if price <= 0:
                self.log(f"Lỗi lấy giá")
                return
                
            step = get_step_size(self.symbol)
            if step <= 0:
                step = 0.001
            
            qty = (usdt_amount * self.lev) / price
            
            if step > 0:
                steps = qty / step
                qty = round(steps) * step
            
            qty = max(qty, 0)
            qty = round(qty, 8)
            
            min_qty = step
            
            if qty < min_qty:
                self.log(f"⚠️ Số lượng quá nhỏ ({qty}), không đặt lệnh")
                return
                
            self.position_attempt_count += 1
            if self.position_attempt_count > self.max_position_attempts:
                self.log(f"⚠️ Đã đạt giới hạn số lần thử mở lệnh ({self.max_position_attempts})")
                self.position_attempt_count = 0
                return
                
            res = place_order(self.symbol, side, qty)
            if not res:
                self.log(f"Lỗi khi đặt lệnh")
                return
                
            executed_qty = float(res.get('executedQty', 0))
            if executed_qty <= 0:
                self.log(f"Lệnh không khớp, số lượng thực thi: {executed_qty}")
                return

            self.entry = float(res.get('avgPrice', price))
            self.side = side
            self.qty = executed_qty if side == "BUY" else -executed_qty
            self.status = "open"
            self.position_open = True
            self.position_attempt_count = 0
            
            logger.info(f"✅ Đã mở vị thế {self.symbol} {side} tại {self.entry:.4f}")

        except Exception as e:
            self.position_open = False
            self.log(f"❌ Lỗi khi vào lệnh: {str(e)}")

    def close_position(self, reason=""):
        """Đóng vị thế với số lượng chính xác"""
        try:
            cancel_all_orders(self.symbol)
            
            if abs(self.qty) > 0:
                close_side = "SELL" if self.side == "BUY" else "BUY"
                close_qty = abs(self.qty)
                
                step = get_step_size(self.symbol)
                if step > 0:
                    steps = close_qty / step
                    close_qty = round(steps) * step
                
                close_qty = max(close_qty, 0)
                close_qty = round(close_qty, 8)
                
                res = place_order(self.symbol, close_side, close_qty)
                if res:
                    price = float(res.get('avgPrice', 0))
                    logger.info(f"⛔ Đã đóng vị thế {self.symbol} tại {price:.4f} - Lý do: {reason}")
                    
                    self.status = "waiting"
                    self.side = ""
                    self.qty = 0
                    self.entry = 0
                    self.position_open = False
                    self.last_trade_time = time.time()
                    self.last_close_time = time.time()
                else:
                    logger.error(f"Lỗi khi đóng lệnh {self.symbol}")
        except Exception as e:
            logger.error(f"❌ Lỗi khi đóng lệnh {self.symbol}: {str(e)}")

# ========== QUẢN LÝ BOT ==========
class BotManager:
    def __init__(self, app):
        self.ws_manager = WebSocketManager()
        self.bots = {}
        self.running = True
        self.start_time = time.time()
        self.app = app
        
        logger.info("🟢 HỆ THỐNG BOT ĐÃ KHỞI ĐỘNG")
        
        # Bắt đầu thread kiểm tra trạng thái
        self.status_thread = threading.Thread(target=self._status_monitor, daemon=True)
        self.status_thread.start()
        
        # Thêm các bot từ cấu hình
        if BOT_CONFIGS:
            for config in BOT_CONFIGS:
                self.add_bot(*config)
        else:
            logger.info("⚠️ Không có cấu hình bot nào được tìm thấy!")
        
        # Thông báo số dư ban đầu
        try:
            balance = get_balance()
            logger.info(f"💰 SỐ DƯ BAN ĐẦU: {balance:.2f} USDT")
        except Exception as e:
            logger.error(f"⚠️ Lỗi lấy số dư ban đầu: {str(e)}")

    def log(self, message):
        logger.info(f"[SYSTEM] {message}")

    def add_bot(self, symbol, lev, percent, tp, sl, indicator):
        symbol = symbol.upper()
        if symbol in self.bots:
            self.log(f"⚠️ Đã có bot cho {symbol}")
            return False
            
        if not API_KEY or not API_SECRET:
            self.log("❌ Chưa cấu hình API Key và Secret Key!")
            return False
            
        try:
            price = get_current_price(symbol)
            if price <= 0:
                self.log(f"❌ Không thể lấy giá cho {symbol}")
                return False
            
            positions = get_positions(symbol)
            if positions and any(float(pos.get('positionAmt', 0)) != 0 for pos in positions):
                self.log(f"⚠️ Đã có vị thế mở cho {symbol} trên Binance")
                return False
            
            bot = IndicatorBot(
                symbol, lev, percent, tp, sl, 
                indicator, self.ws_manager
            )
            self.bots[symbol] = bot
            self.log(f"✅ Đã thêm bot: {symbol} | ĐB: {lev}x | %: {percent} | TP/SL: {tp}%/{sl}%")
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot {symbol}: {str(e)}")
            return False

    def stop_bot(self, symbol):
        symbol = symbol.upper()
        bot = self.bots.get(symbol)
        if bot:
            bot.stop()
            if bot.status == "open":
                bot.close_position("⛔ Dừng bot thủ công")
            self.log(f"⛔ Đã dừng bot cho {symbol}")
            del self.bots[symbol]
            return True
        return False

    def stop_all(self):
        self.log("⛔ Đang dừng tất cả bot...")
        for symbol in list(self.bots.keys()):
            self.stop_bot(symbol)
        self.ws_manager.stop()
        self.running = False
        self.log("🔴 Hệ thống đã dừng")

    def _status_monitor(self):
        """Kiểm tra và ghi log trạng thái định kỳ"""
        while self.running:
            try:
                uptime = time.time() - self.start_time
                hours, rem = divmod(uptime, 3600)
                minutes, seconds = divmod(rem, 60)
                uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                
                active_bots = [s for s, b in self.bots.items() if not b._stop]
                balance = get_balance()
                
                status_msg = (
                    f"📊 BÁO CÁO HỆ THỐNG\n"
                    f"⏱ Thời gian hoạt động: {uptime_str}\n"
                    f"🤖 Số bot đang chạy: {len(active_bots)}\n"
                    f"📈 Bot hoạt động: {', '.join(active_bots) if active_bots else 'Không có'}\n"
                    f"💰 Số dư khả dụng: {balance:.2f} USDT"
                )
                logger.info(status_msg)
                
                for symbol, bot in self.bots.items():
                    if bot.status == "open":
                        status_msg = (
                            f"🔹 {symbol}\n"
                            f"📌 Hướng: {bot.side}\n"
                            f"🏷️ Giá vào: {bot.entry:.4f}\n"
                            f"📊 Khối lượng: {abs(bot.qty)}\n"
                            f"⚖️ Đòn bẩy: {bot.lev}x\n"
                            f"🎯 TP: {bot.tp}% | 🛡️ SL: {bot.sl}%"
                        )
                        logger.info(status_msg)
                
            except Exception as e:
                logger.error(f"Lỗi báo cáo trạng thái: {str(e)}")
            
            time.sleep(6 * 3600)

# Khởi tạo BotManager sau khi app được tạo
bot_manager = None

@app.before_first_request
def initialize_bot_manager():
    global bot_manager
    bot_manager = BotManager(app)

# ========== ROUTES CHO GIAO DIỆN WEB ==========
@app.route('/')
def index():
    """Trang chủ hiển thị trạng thái bot"""
    if not bot_manager:
        return "Hệ thống đang khởi động, vui lòng thử lại sau..."
    
    # Tính thời gian hoạt động
    uptime = time.time() - bot_manager.start_time
    hours, rem = divmod(uptime, 3600)
    minutes, seconds = divmod(rem, 60)
    uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    # Lấy số dư tài khoản
    balance = get_balance()
    
    # Lấy danh sách bot
    bots = []
    for symbol, bot in bot_manager.bots.items():
        bots.append({
            'symbol': symbol,
            'leverage': bot.lev,
            'percent': bot.percent,
            'tp': bot.tp,
            'sl': bot.sl,
            'status': bot.status,
            'side': bot.side,
            'entry': bot.entry,
            'qty': bot.qty,
            'indicator': bot.indicator
        })
    
    # Lấy vị thế đang mở
    positions = get_positions()
    open_positions = []
    for pos in positions:
        position_amt = float(pos.get('positionAmt', 0))
        if position_amt != 0:
            open_positions.append({
                'symbol': pos.get('symbol', ''),
                'side': "LONG" if position_amt > 0 else "SHORT",
                'amount': abs(position_amt),
                'entry': float(pos.get('entryPrice', 0)),
                'pnl': float(pos.get('unRealizedProfit', 0))
            })
    
    return render_template(
        'index.html',
        bots=bots,
        balance=balance,
        uptime=uptime_str,
        open_positions=open_positions,
        active_bots_count=len(bot_manager.bots)
    )

@app.route('/add_bot', methods=['GET', 'POST'])
def add_bot():
    """Thêm bot mới"""
    if request.method == 'POST':
        symbol = request.form.get('symbol', '').strip().upper()
        leverage = int(request.form.get('leverage', 20))
        percent = float(request.form.get('percent', 5))
        tp = float(request.form.get('tp', 10))
        sl = float(request.form.get('sl', 5))
        indicator = "RSI"
        
        if bot_manager.add_bot(symbol, leverage, percent, tp, sl, indicator):
            return redirect(url_for('index'))
        else:
            return "Lỗi khi thêm bot, vui lòng kiểm tra log", 400
    
    # Nếu là GET, hiển thị form
    return render_template('add_bot.html')

@app.route('/stop_bot/<symbol>')
def stop_bot(symbol):
    """Dừng bot theo symbol"""
    if bot_manager.stop_bot(symbol):
        return redirect(url_for('index'))
    else:
        return "Không tìm thấy bot", 404

@app.route('/stop_all')
def stop_all():
    """Dừng tất cả bot"""
    bot_manager.stop_all()
    return redirect(url_for('index'))

@app.route('/balance')
def balance():
    """API trả về số dư tài khoản"""
    return jsonify({'balance': get_balance()})

@app.route('/positions')
def positions():
    """API trả về vị thế đang mở"""
    positions = get_positions()
    open_positions = []
    for pos in positions:
        position_amt = float(pos.get('positionAmt', 0))
        if position_amt != 0:
            open_positions.append({
                'symbol': pos.get('symbol', ''),
                'side': "LONG" if position_amt > 0 else "SHORT",
                'amount': abs(position_amt),
                'entry': float(pos.get('entryPrice', 0)),
                'pnl': float(pos.get('unRealizedProfit', 0))
            })
    return jsonify(open_positions)

# ========== CHẠY ỨNG DỤNG ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)