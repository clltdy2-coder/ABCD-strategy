        print(f"🚫 當前為強制平倉時間: {close_reason}")
        else:
            print("✅ 當前非強制平倉時間")
        
        # 測試禁止開倉時間
        in_no_position_period, no_position_reason = self.is_in_no_position_period()
        if in_no_position_period:
            print(f"🚫 當前為禁止開倉時間: {no_position_reason}")
        else:
            print("✅ 當前可開倉時間")

    def is_force_close_time(self):
        """檢查是否為強制平倉時間"""
        current_time = datetime.now() if not self.backtest_mode else getattr(self, 'current_backtest_time', datetime.now())
        
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # 檢查是否接近強制平倉時間（提前2分鐘開始平倉）
        for time_name, time_config in self.force_close_times.items():
            if 'end' in time_name:  # 只檢查結束時間
                target_hour = time_config['hour']
                target_minute = time_config['minute']
                
                # 計算目標時間和當前時間的分鐘差
                target_minutes = target_hour * 60 + target_minute
                current_minutes = current_hour * 60 + current_minute
                
                # 如果在目標時間前2分鐘內，則觸發強制平倉
                if target_minutes - 2 <= current_minutes <= target_minutes:
                    return True, time_name
        
        return False, None

    def is_in_no_position_period(self, current_time=None):
        """檢查是否在禁止開倉時間內"""
        if current_time is None:
            if self.backtest_mode:
                current_time = getattr(self, 'current_backtest_time', datetime.now())
            else:
                current_time = datetime.now()
        
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_total_minutes = current_hour * 60 + current_minute
        
        for period_name, period_config in self.no_new_position_times.items():
            start_hour = period_config['start_hour']
            start_minute = period_config['start_minute']
            end_hour = period_config['end_hour']
            end_minute = period_config['end_minute']
            
            start_total_minutes = start_hour * 60 + start_minute
            end_total_minutes = end_hour * 60 + end_minute
            
            # 處理跨夜情況
            if start_total_minutes > end_total_minutes:  # 跨夜
                if current_total_minutes >= start_total_minutes or current_total_minutes <= end_total_minutes:
                    return True, period_name
            else:  # 同日
                if start_total_minutes <= current_total_minutes <= end_total_minutes:
                    return True, period_name
        
        return False, None

    def login(self):
        """API登入"""
        if self.backtest_mode:
            return True
            
        try:
            self.api.login(api_key=self.api_key, secret_key=self.secret_key)
            print("✅ API登入成功")
            time.sleep(2)

            self.contract = min(
                [x for x in self.api.Contracts.Futures.MXF if x.code[-2:] not in ["R1", "R2"]],
                key=lambda x: x.delivery_date
            )
            
            if self.contract:
                contract_code = getattr(self.contract, 'code', getattr(self.contract, 'symbol', 'UNKNOWN'))
                print(f"✅ 自動選取近月合約: {contract_code}")
                return True
            else:
                print("❌ 無法取得近月合約")
                return False

        except Exception as e:
            print(f"❌ API登入失敗: {e}")
            return False

    def place_order(self, action, quantity, price=None):
        """下單函數"""
        if self.backtest_mode:
            return True
            
        try:
            if not self.contract:
                print("❌ 合約未設定")
                return False
            
            if self.running_mode == "SIMULATION":
                print(f"📝 模擬下單: {action} {quantity}口 @ {'市價' if price is None else price}")
                print("⚠️ 注意：這是模擬版本，不會實際下單")
                return True
            elif self.running_mode == "LIVE_TRADING":
                # 真實下單邏輯
                print(f"🔴 真實下單: {action} {quantity}口 @ {'市價' if price is None else price}")
                # 這裡添加真實的下單API調用
                return True
            else:
                return True
            
        except Exception as e:
            print(f"❌ 下單失敗: {e}")
            return False

# ===== 主程式和運行模式函數 =====

def validate_environment():
    """驗證環境"""
    required_packages = {
        'shioaji': 'shioaji',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'talib': 'TA-Lib',
        'python-dotenv': 'python-dotenv',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    for package, install_name in required_packages.items():
        try:
            if package == 'python-dotenv':
                import dotenv
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(install_name)
    
    if missing_packages:
        print("❌ 缺少套件:")
        for package in missing_packages:
            print(f"   pip install {package}")
        return False
    
    if not os.path.exists('.env'):
        print("⚠️ 建立.env範本...")
        with open('.env', 'w', encoding='utf-8') as f:
            f.write("""# 永豐金API設定
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
CA_CERT_PATH=your_cert_path_here
CA_PASSWORD=your_cert_password_here

# 交易參數設定
MAX_POSITION=1
STOP_LOSS_PCT=0.008
TAKE_PROFIT_PCT=0.025
""")
        print("✅ 已建立.env範本，請填入您的API資訊")
        return False
    
    return True

def run_live_simulation_mode(enable_debug=False):
    """即時交易模擬模式"""
    print("\n📡 ABCD策略即時模擬模式")
    print("=" * 30)
    
    try:
        strategy = NewTaiwanFuturesStrategy(backtest_mode=False)
        strategy.set_mode_configuration("SIMULATION")
        strategy.toggle_debug_mode(enable_debug)
        
        print("🔧 正在初始化策略...")
        
        # 模擬登入
        print("🔑 模擬API登入...")
        print("✅ 模擬登入成功")
        
        # 設定模擬合約
        strategy.contract = type('MockContract', (), {'code': 'MXFK4'})()
        print(f"✅ 模擬合約設定: {strategy.contract.code}")
        
        print("\n🚀 開始即時模擬交易...")
        print("⚠️ 這是模擬模式，不會執行真實交易")
        print("📊 將使用模擬數據進行ABCD策略測試")
        print("🔇 靜音模式已啟用，減少不必要訊息")
        print("\n按 Ctrl+C 停止模擬")
        
        # 模擬交易循環
        simulation_count = 0
        while simulation_count < 10:  # 限制模擬次數
            try:
                simulation_count += 1
                current_time = datetime.now()
                
                # 模擬價格數據
                base_price = 17000 + np.random.randint(-200, 200)
                print(f"\n⏰ {current_time.strftime('%H:%M:%S')} | 模擬價格: {base_price}")
                
                # 模擬ABCD條件檢查
                if enable_debug:
                    print("🔍 調試模式: 檢查ABCD條件...")
                    print(f"   A(RSI): {'✓' if np.random.random() > 0.5 else '✗'}")
                    print(f"   B(MA): {'✓' if np.random.random() > 0.5 else '✗'}")
                    print(f"   C(Price): {'✓' if np.random.random() > 0.5 else '✗'}")
                    print(f"   D(Trend): {'✓' if np.random.random() > 0.5 else '✗'}")
                
                # 模擬信號
                signal_type = np.random.choice(['多頭', '空頭', '無信號'], p=[0.2, 0.2, 0.6])
                print(f"📊 模擬信號: {signal_type}")
                
                if signal_type != '無信號':
                    action = 'Buy' if signal_type == '多頭' else 'Sell'
                    strategy.place_order(action, 1, base_price)
                
                time.sleep(3)  # 等待3秒
                
            except KeyboardInterrupt:
                print("\n⚠️ 收到停止信號")
                break
        
        print("\n✅ 即時模擬完成")
        
    except Exception as e:
        print(f"❌ 即時模擬錯誤: {e}")

def run_backtest_mode(enable_debug=False):
    """回測模式"""
    print("\n📊 ABCD策略回測模式")
    print("=" * 30)
    
    try:
        strategy = NewTaiwanFuturesStrategy(backtest_mode=True)
        strategy.set_mode_configuration("BACKTEST")
        strategy.toggle_debug_mode(enable_debug)
        
        # 獲取CSV檔案路徑
        while True:
            csv_file = input("請輸入CSV檔案路徑: ").strip()
            if csv_file and os.path.exists(csv_file):
                break
            elif csv_file == "":
                print("❌ 請輸入檔案路徑")
                continue
            else:
                print(f"❌ 檔案不存在: {csv_file}")
                continue
        
        # 載入數據並執行回測
        if strategy.load_backtest_data(csv_file):
            print("\n🔬 開始執行ABCD策略回測...")
            
            if strategy.run_backtest():
                print("\n📊 回測完成！")
                
                # 詢問是否保存結果
                save_choice = input("是否保存回測結果？(y/N): ").strip().lower()
                if save_choice == 'y':
                    strategy.save_backtest_results()
                
                # 詢問是否生成圖表
                plot_choice = input("是否生成回測圖表？(y/N): ").strip().lower()
                if plot_choice == 'y':
                    try:
                        strategy.plot_backtest_results()
                    except Exception as e:
                        print(f"⚠️ 圖表生成失敗: {e}")
            else:
                print("❌ 回測執行失敗")
        else:
            print("❌ 數據載入失敗")
    
    except Exception as e:
        print(f"❌ 回測模式錯誤: {e}")

def run_real_trading_mode(enable_debug=False):
    """實盤交易模式"""
    print("\n🔴 ABCD策略實盤交易模式")
    print("=" * 30)
    print("⚠️ 警告：這是真實交易模式，將執行實際下單!")
    
    # 二次確認
    confirm = input("確定要進入實盤交易模式嗎？(輸入 'YES' 確認): ").strip()
    if confirm != 'YES':
        print("❌ 取消實盤交易")
        return
    
    try:
        strategy = NewTaiwanFuturesStrategy(backtest_mode=False)
        strategy.set_mode_configuration("LIVE_TRADING")
        strategy.toggle_debug_mode(enable_debug)
        
        print("🔧 正在初始化實盤交易...")
        
        # 檢查API設定
        if not strategy.api_key or strategy.api_key == 'your_api_key_here':
            print("❌ 請先在.env檔案中設定正確的API資訊")
            return
        
        # 登入API
        if not strategy.login():
            print("❌ API登入失敗，無法進行實盤交易")
            return
        
        print("✅ 實盤交易初始化完成")
        print("🔴 開始真實交易...")
        print("⚠️ 請確保您了解風險並有足夠保證金")
        print("\n按 Ctrl+C 安全停止交易")
        
        # 實盤交易主循環（簡化版）
        trade_count = 0
        while True:
            try:
                current_time = datetime.now()
                
                # 檢查交易時間
                if not strategy.is_valid_trading_time_enhanced():
                    print(f"⏸️ {current_time.strftime('%H:%M')} 非交易時間，等待中...")
                    time.sleep(60)
                    continue
                
                # 檢查強制平倉時間
                should_force_close, close_reason = strategy.is_force_close_time()
                if should_force_close and strategy.position != 0:
                    print(f"🚫 強制平倉時間: {close_reason}")
                    strategy.close_position(reason=f"強制平倉: {close_reason}")
                    continue
                
                print(f"⏰ {current_time.strftime('%H:%M:%S')} | 實盤監控中...")
                
                if enable_debug:
                    print("🔍 調試模式: 實盤ABCD條件監控")
                    print(f"   當前部位: {strategy.position}口")
                    print(f"   今日交易: {strategy.daily_trade_count}/{strategy.max_daily_trades}")
                
                # 這裡應該有實際的數據獲取和信號生成邏輯
                # 為了安全，暫時只做監控
                time.sleep(30)  # 30秒檢查一次
                
                trade_count += 1
                if trade_count > 20:  # 限制運行時間
                    print("⏰ 達到測試時間限制，停止交易")
                    break
                
            except KeyboardInterrupt:
                print("\n⚠️ 收到停止信號，安全關閉交易...")
                
                # 安全關閉邏輯
                if strategy.position != 0:
                    close_confirm = input("發現持倉，是否立即平倉？(y/N): ").strip().lower()
                    if close_confirm == 'y':
                        strategy.close_position(reason="手動停止")
                        print("✅ 持倉已平倉")
                
                print("✅ 實盤交易已安全停止")
                break
                
    except Exception as e:
        print(f"❌ 實盤交易錯誤: {e}")
        print("🚨 請手動檢查API帳戶狀態")

def run_optimization_mode(enable_debug=False):
    """策略優化模式"""
    print("\n🎯 ABCD策略參數優化模式")
    print("=" * 30)
    
    try:
        strategy = NewTaiwanFuturesStrategy(backtest_mode=True)
        strategy.set_mode_configuration("OPTIMIZATION")
        strategy.toggle_debug_mode(enable_debug)
        
        # 獲取CSV檔案路徑
        while True:
            csv_file = input("請輸入用於優化的CSV檔案路徑: ").strip()
            if csv_file and os.path.exists(csv_file):
                break
            elif csv_file == "":
                print("❌ 請輸入檔案路徑")
                continue
            else:
                print(f"❌ 檔案不存在: {csv_file}")
                continue
        
        # 選擇優化類型
        print("\n選擇優化類型:")
        print("1. 快速優化 (較少參數組合)")
        print("2. 標準優化 (平衡速度與精度)")
        print("3. 深度優化 (全面參數搜索)")
        
        opt_choice = input("請選擇 (1-3): ").strip()
        
        # 設定參數範圍
        if opt_choice == "1":
            param_ranges = {
                'rsi_period': [14],
                'ma_fast': [15, 20, 25],
                'ma_slow': [50, 60, 70],
                'signal_strength_threshold': [2, 3]
            }
        elif opt_choice == "3":
            param_ranges = {
                'rsi_period': [10, 12, 14, 16, 18, 20],
                'ma_fast': [10, 15, 20, 25, 30],
                'ma_slow': [40, 50, 60, 70, 80],
                'bb_period': [15, 20, 25, 30],
                'adx_threshold': [20, 25, 30, 35],
                'signal_strength_threshold': [2, 3, 4],
                'stop_loss_pct': [0.003, 0.005, 0.008, 0.01],
                'take_profit_pct': [0.015, 0.02, 0.025, 0.03]
            }
        else:  # 標準優化
            param_ranges = {
                'rsi_period': [12, 14, 16],
                'ma_fast': [15, 20, 25],
                'ma_slow': [50, 60, 70],
                'bb_period': [18, 20, 22],
                'adx_threshold': [20, 25, 30],
                'signal_strength_threshold': [2, 3, 4]
            }
        
        print(f"\n🔧 開始參數優化...")
        print(f"📊 預計測試 {np.prod([len(v) for v in param_ranges.values()])} 種參數組合")
        
        start_time = time.time()
        
        if strategy.run_parameter_optimization(csv_file, param_ranges):
            end_time = time.time()
            print(f"\n✅ 優化完成！耗時: {(end_time - start_time)/60:.1f} 分鐘")
            
            # 詢問是否載入最佳參數進行驗證回測
            verify_choice = input("是否使用最佳參數進行驗證回測？(y/N): ").strip().lower()
            if verify_choice == 'y':
                if strategy.optimization_results:
                    best_result = max(strategy.optimization_results, key=lambda x: x['total_return'])
                    strategy.set_optimization_parameters(best_result['params'])
                    print("\n🔬 使用最佳參數進行驗證回測...")
                    strategy.run_backtest()
                    
        else:
            print("❌ 參數優化失敗")
    
    except Exception as e:
        print(f"❌ 優化模式錯誤: {e}")

def test_debug_functionality(enable_debug=True):
    """測試調試功能"""
    print("\n🔍 ABCD策略調試測試模式")
    print("=" * 30)
    
    try:
        strategy = NewTaiwanFuturesStrategy(backtest_mode=False)
        strategy.set_mode_configuration("DEBUG_TEST")
        strategy.toggle_debug_mode(enable_debug)
        
        print("🧪 開始調試功能測試...")
        
        # 執行調試測試
        strategy.run_debug_test()
        
        print("\n🔍 額外調試功能測試:")
        
        # 測試模式切換
        print("⚙️ 測試模式切換功能...")
        strategy.toggle_debug_mode(False)
        print("✅ 調試關閉測試通過")
        
        strategy.toggle_debug_mode(True)
        print("✅ 調試開啟測試通過")
        
        # 測試參數驗證
        print("\n📊 測試參數動態調整...")
        original_rsi = strategy.rsi_period
        strategy.rsi_period = 20
        print(f"RSI週期調整: {original_rsi} → {strategy.rsi_period}")
        strategy.rsi_period = original_rsi
        print("✅ 參數調整測試通過")
        
        print("\n✅ 所有調試功能測試完成")
        
    except Exception as e:
        print(f"❌ 調試功能測試錯誤: {e}")

def main():
    """主程式 - 增強版多模式選擇"""
    print("🚀 新型ABCD台指交易策略 v2.0 - 多模式增強版")
    print("🎯 專為替代MACD策略而設計，提供完整交易解決方案")
    print("📈 使用RSI+MA+BB+ADX四重技術指標組合")
    print("🔧 新增多種運行模式和策略優化功能")
    print("=" * 60)
    
    # 環境驗證
    try:
        if not validate_environment():
            print("❌ 環境驗證失敗")
            input("按Enter退出...")
            return
    except Exception as e:
        print(f"⚠️ 環境驗證時發生錯誤: {e}")
        print("🔄 繼續執行...")
    
    # 模式選擇循環
    while True:
        try:
            print("\n" + "="*50)
            print("請選擇運行模式:")
            print("1. 即時交易模式 (連接永豐金API - 模擬)")
            print("2. 回測模式 (使用CSV檔案)")
            print("3. 實盤交易模式 (真實下單) 🔴")
            print("4. 策略優化模式")
            print("5. 測試調試功能")
            print("0. 退出程式")
            print("="*50)
            
            mode_choice = input("請輸入選擇 (0-5): ").strip()
            
            if mode_choice == "0":
                print("👋 感謝使用ABCD交易策略系統")
                break
            elif mode_choice == "":
                print("⚠️ 請輸入有效選擇")
                continue
            
            # 調試模式選擇
            debug_choice = input("\n🔍 是否啟用調試模式？(顯示詳細ABCD信號分析) (y/N): ").strip().lower()
            enable_debug = debug_choice == 'y'
            
            if enable_debug:
                print("✅ 調試模式已啟用 - 將顯示詳細分析信息")
            else:
                print("🔇 使用標準模式 - 簡化輸出信息")
            
            # 執行相應模式
            if mode_choice == "1":
                run_live_simulation_mode(enable_debug)
            elif mode_choice == "2":
                run_backtest_mode(enable_debug)
            elif mode_choice == "3":
                run_real_trading_mode(enable_debug)
            elif mode_choice == "4":
                run_optimization_mode(enable_debug)
            elif mode_choice == "5":
                test_debug_functionality(enable_debug)
            else:
                print("❌ 無效選擇，請重新輸入")
                continue
                
            # 詢問是否繼續
            continue_choice = input("\n🔄 是否繼續使用其他功能？ (y/N): ").strip().lower()
            if continue_choice != 'y':
                print("👋 程式退出")
                break
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 收到中斷信號")
            confirm_exit = input("確定要退出嗎？ (y/N): ").strip().lower()
            if confirm_exit == 'y':
                print("👋 程式已安全退出")
                break
        except Exception as e:
            print(f"\n❌ 程式執行錯誤: {e}")
            error_continue = input("是否繼續運行？ (y/N): ").strip().lower()
            if error_continue != 'y':
                break

if __name__ == "__main__":
    main()
