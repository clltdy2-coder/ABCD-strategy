#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ABCD策略交易系統 v2.0 - 擴充版
支援多種運行模式：
1. 即時交易模式 (模擬)
2. 回測模式
3. 實盤交易模式
4. 策略優化模式
5. 測試調試功能
"""

import shioaji as sj
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
import talib
import threading
from collections import deque
from dotenv import load_dotenv
import warnings
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from itertools import product
import pickle

warnings.filterwarnings('ignore')

# 關閉 shioaji 預設的 Info/Debug 輸出
logging.getLogger("shioaji").setLevel(logging.WARNING)

# 載入環境變數
load_dotenv()

class NewTaiwanFuturesStrategy:
    """
    新型台指交易策略 - ABCD條件版 (擴充版)
    包含多種運行模式和策略優化功能
    """
    
    def __init__(self, backtest_mode=False, real_trading=False):
        """
        初始化策略
        
        Args:
            backtest_mode: 是否為回測模式
            real_trading: 是否為實盤交易模式
        """
        self.backtest_mode = backtest_mode
        self.real_trading = real_trading
        
        # 初始化運行模式
        self.mode = "simulation"  # 默認為模擬模式
        if backtest_mode:
            self.mode = "backtest"
        elif real_trading:
            self.mode = "real_trading"
        
        # 初始化 silent_mode 屬性
        self.silent_mode = True
        self.suppress_tick_messages = True
        self.debug_mode = False
        
        if not backtest_mode:
            self.api = sj.Shioaji()
            
            # 從環境變數取得API資訊
            self.api_key = os.getenv('API_KEY')
            self.secret_key = os.getenv('SECRET_KEY')
            self.ca_path = os.getenv('CA_CERT_PATH')
            self.ca_password = os.getenv('CA_PASSWORD')
            
            if not self.api_key or not self.secret_key:
                print("⚠️ API憑證未設定，將使用模擬模式")
        
        # 策略參數設定
        self.max_position = int(os.getenv('MAX_POSITION', '1'))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '0.005'))
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', '0.025'))
        
        # 移動止利設定
        self.trailing_profit_threshold = 150
        self.trailing_stop_distance = 40
        self.is_trailing_active = False
        self.trailing_high_price = 0
        self.trailing_low_price = 0
        
        # ABCD策略參數
        self.init_abcd_parameters()
        
        # 風險管理參數
        self.init_risk_management()
        
        # 時間管理參數
        self.init_time_management()
        
        # 交易狀態變數
        self.init_trading_state()
        
        # 統計變數
        self.init_statistics()
        
        # 回測專用變數
        self.init_backtest_variables()
        
        # 優化模式變數
        self.optimization_results = []
        self.best_parameters = None
        
        print(f"🚀 ABCD策略初始化完成 - 模式: {self.mode}")
        self.display_parameters()
    
    def init_abcd_parameters(self):
        """初始化ABCD策略參數"""
        # A條件: RSI指標
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # B條件: 移動平均線
        self.ma_fast = 20
        self.ma_slow = 60
        
        # C條件: 布林通道
        self.bb_period = 20
        self.bb_std = 2.0
        self.price_position_threshold = 0.3
        
        # D條件: ADX趨勢指標
        self.adx_period = 14
        self.adx_threshold = 25
        self.di_threshold = 10
        
        # 綜合信號設定
        self.signal_strength_threshold = 3
        self.volume_threshold = 1.2
    
    def init_risk_management(self):
        """初始化風險管理參數"""
        self.max_daily_trades = 3
        self.min_signal_interval = 1800
        self.position_timeout = 14400
        self.max_consecutive_losses = 2
    
    def init_time_management(self):
        """初始化時間管理參數"""
        self.avoid_open_close_minutes = 30
        self.lunch_break_avoid = True
        
        # 不留倉策略設定
        self.force_close_times = {
            'morning_session_end': {'hour': 13, 'minute': 20},
            'night_session_end': {'hour': 23, 'minute': 50},
            'session_start_clear_morning': {'hour': 8, 'minute': 45},
            'session_start_clear_afternoon': {'hour': 15, 'minute': 0}
        }
        
        self.no_new_position_times = {
            'before_morning_close': {'start_hour': 13, 'start_minute': 15, 'end_hour': 15, 'end_minute': 5},
            'before_night_close': {'start_hour': 23, 'start_minute': 45, 'end_hour': 8, 'end_minute': 50}
        }
        
        self.no_overnight_mode = True
        self.last_clear_time = None
    
    def init_trading_state(self):
        """初始化交易狀態"""
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self.highest_profit_price = 0
        self.lowest_profit_price = 0
        self.last_signal_time = None
        self.daily_trade_count = 0
        self.consecutive_losses = 0
        self.last_trade_profit = 0
        self.data_queue = deque(maxlen=200)
        self.tick_data = deque(maxlen=100)
    
    def init_statistics(self):
        """初始化統計變數"""
        self.trade_count = 0
        self.win_count = 0
        self.total_profit = 0
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.market_volatility = 0
        self.trend_strength = 0
        self.market_condition = "UNKNOWN"
    
    def init_backtest_variables(self):
        """初始化回測變數"""
        self.backtest_data = None
        self.backtest_results = []
        self.backtest_equity_curve = []
        self.backtest_trades = []
        self.backtest_daily_returns = []
        
        if not self.backtest_mode:
            self.contract = None
            self.last_status_time = datetime.now()
            self.status_display_interval = 300
            self.last_price_update = 0
            self.price_update_count = 0
    
    def display_parameters(self):
        """顯示策略參數"""
        print(f"📊 策略參數:")
        print(f"   RSI({self.rsi_period}), MA({self.ma_fast}/{self.ma_slow})")
        print(f"   BB({self.bb_period}), ADX({self.adx_period})")
        print(f"🎯 信號條件: 需滿足{self.signal_strength_threshold}/4個ABCD條件")
        print(f"🛡️ 風險控管: 每日最大{self.max_daily_trades}筆")
        print(f"🚫 不留倉設定: 13:20和23:50強制平倉")
    
    # ========== 主程式運行控制 ==========
    
    def run_mode_selector(self):
        """
        主程式模式選擇器
        提供所有運行模式的統一入口
        """
        while True:
            try:
                self.display_mode_menu()
                mode_choice = input("\n請輸入選擇 (0-5): ").strip()
                
                if mode_choice == "0":
                    print("👋 程式退出")
                    break
                    
                # 詢問是否啟用調試模式
                debug_choice = input("🔍 是否啟用調試模式？(y/N): ").strip().lower()
                self.debug_mode = debug_choice == 'y'
                
                # 執行選擇的模式
                success = self.execute_mode(mode_choice)
                
                if not success:
                    print("❌ 模式執行失敗")
                
                # 詢問是否繼續
                if not self.ask_continue():
                    break
                    
            except KeyboardInterrupt:
                print("\n⚠️ 收到中斷信號")
                if self.confirm_exit():
                    break
            except Exception as e:
                print(f"❌ 錯誤: {e}")
                if not self.handle_error():
                    break
    
    def display_mode_menu(self):
        """顯示模式選單"""
        print("\n" + "="*50)
        print("🚀 ABCD策略交易系統 v2.0")
        print("="*50)
        print("請選擇運行模式:")
        print("1. 即時交易模式 (連接永豐金API - 模擬)")
        print("2. 回測模式 (使用CSV檔案)")
        print("3. 實盤交易模式 (真實下單) 🔴")
        print("4. 策略優化模式")
        print("5. 測試調試功能")
        print("0. 退出程式")
    
    def execute_mode(self, mode_choice):
        """
        執行選擇的模式
        
        Args:
            mode_choice: 模式選擇 (字串)
            
        Returns:
            bool: 執行是否成功
        """
        mode_map = {
            "1": self.run_simulation_mode,
            "2": self.run_backtest_mode,
            "3": self.run_real_trading_mode,
            "4": self.run_optimization_mode,
            "5": self.run_debug_test_mode
        }
        
        if mode_choice in mode_map:
            return mode_map[mode_choice]()
        else:
            print("❌ 無效選擇")
            return False
    
    # ========== 以下是各個模式的簡化實現 ==========
    
    def run_simulation_mode(self):
        """即時交易模擬模式"""
        print("\n📡 進入即時交易模擬模式")
        print("⚠️ 注意: 這是模擬模式，不會真實下單")
        print("功能開發中...")
        return True
    
    def run_backtest_mode(self):
        """回測模式"""
        print("\n📈 進入回測模式")
        print("請準備CSV檔案進行回測")
        print("功能開發中...")
        return True
    
    def run_real_trading_mode(self):
        """實盤交易模式"""
        print("\n🔴 進入實盤交易模式")
        print("⚠️ 警告: 這是實盤模式，將執行真實交易！")
        confirm = input("\n確定要進入實盤模式嗎？輸入'YES'確認: ").strip()
        if confirm != 'YES':
            print("已取消")
            return False
        print("功能開發中...")
        return True
    
    def run_optimization_mode(self):
        """策略優化模式"""
        print("\n🔧 進入策略優化模式")
        print("選擇優化類型:")
        print("1. 參數優化 (Grid Search)")
        print("2. 前進分析 (Walk Forward)")
        print("3. 蒙地卡羅模擬 (Monte Carlo)")
        print("功能開發中...")
        return True
    
    def run_debug_test_mode(self):
        """測試調試模式"""
        print("\n🔍 進入測試調試模式")
        print("測試功能:")
        print("1. 測試技術指標計算")
        print("2. 測試信號生成")
        print("3. 測試風險管理")
        print("4. 測試時間管理")
        print("5. 測試訂單執行")
        print("功能開發中...")
        return True
    
    def ask_continue(self):
        """詢問是否繼續"""
        choice = input("\n🔄 是否繼續使用其他功能？(y/N): ").strip().lower()
        return choice == 'y'
    
    def confirm_exit(self):
        """確認退出"""
        choice = input("確定要退出嗎？(y/N): ").strip().lower()
        return choice == 'y'
    
    def handle_error(self):
        """處理錯誤"""
        choice = input("是否繼續運行？(y/N): ").strip().lower()
        return choice == 'y'


# ========== 主程式 ==========

def main():
    """主程式入口"""
    print("🚀 ABCD策略交易系統 v2.0 - 擴充版")
    print("="*50)
    
    try:
        # 創建策略實例
        strategy = NewTaiwanFuturesStrategy()
        
        # 執行模式選擇器
        strategy.run_mode_selector()
        
    except KeyboardInterrupt:
        print("\n👋 程式已終止")
    except Exception as e:
        print(f"❌ 程式錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
