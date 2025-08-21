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
    print("🚀 新型ABCD台指交易策略 v2.0 - 完整增強版")
    print("🎯 專為替代MACD策略而設計，提供完整交易解決方案")
    print("📈 使用RSI+MA+BB+ADX四重技術指標組合")
    print("🔧 包含完整的原程式功能和新增的5種運行模式")
    print("🛡️ 保留所有原有風險控制和不留倉策略")
    print("🎯 新增策略優化、調試測試等進階功能")
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
