# 🌟 綜合型科學計算MCP服務器

## 📊 服務器概述

這是一個功能豐富的**多領域科學計算MCP服務器**，最初作為氣象服務器開始，現已演進為一個強大的**學術和科學工具平台**。它整合了數學、密碼學、物理學和天氣預報功能，為研究者、學生和專業人員提供全方位的計算工具。

## 🛠️ 核心功能模組

### 🌤️ **氣象模組** (原始功能)
- **天氣預報**: 根據經緯度獲取詳細天氣預報
- **天氣警報**: 查詢美國各州的活躍天氣警報
- **圖像處理**: 創建天氣圖表縮略圖

### 🧮 **數學計算模組** (基於SymPy)
- **表達式操作**: 簡化、展開、因式分解
- **微積分**: 求導、積分（定積分/不定積分）
- **方程求解**: 代數方程、極限計算
- **LaTeX輸出**: 專業數學格式化

### 🔐 **密碼學模組** (基於SymPy Crypto)
- **古典密碼**: 凱撒密碼、維吉尼亞密碼、仿射密碼
- **現代加密**: RSA非對稱加密、密鑰生成
- **特殊編碼**: 摩斯密碼、柵欄密碼、替換密碼
- **完整生態**: 每種密碼都有對應的加密/解密工具

### ⚛️ **物理學模組** (基於SymPy Physics)
- **經典力學**: 動能、位能、動量、力的計算
- **波動光學**: 波性質、光子能量、薄透鏡公式
- **量子力學**: 量子位元狀態、Pauli閘、對易子計算
- **單位系統**: 國際單位制轉換

## 🚀 安裝與使用

### 安裝依賴
```bash
pip install sympy pillow httpx
```

### 運行服務器
```bash
python main.py
```

### 依賴套件
- `sympy`: 符號數學計算
- `Pillow`: 圖像處理
- `httpx`: HTTP客戶端
- `mcp-server-time`: 時間服務

## 📖 工具列表

### 🧮 數學工具
- `simplify_expression` - 簡化數學表達式
- `differentiate_expression` - 求導數
- `integrate_expression` - 計算積分
- `solve_equation` - 解方程
- `calculate_limit` - 計算極限
- `factor_expression` - 因式分解
- `expand_expression` - 展開表達式

### 🔐 密碼學工具
- `encrypt_shift_cipher` / `decrypt_shift_cipher` - 凱撒密碼
- `encrypt_vigenere_cipher` / `decrypt_vigenere_cipher` - 維吉尼亞密碼
- `encrypt_affine_cipher` / `decrypt_affine_cipher` - 仿射密碼
- `encrypt_railfence_cipher` / `decrypt_railfence_cipher` - 柵欄密碼  
- `generate_rsa_keys` / `encrypt_rsa` / `decrypt_rsa` - RSA加密
- `encode_morse_code` / `decode_morse_code` - 摩斯密碼
- `encrypt_substitution_cipher` - 替換密碼

### ⚛️ 物理學工具
- `convert_units` - 單位轉換
- `calculate_kinetic_energy` - 動能計算
- `calculate_potential_energy` - 位能計算
- `calculate_force` - 力的計算（牛頓第二定律）
- `calculate_momentum` - 動量計算
- `calculate_wave_properties` - 波性質計算
- `calculate_photon_energy` - 光子能量計算
- `thin_lens_calculation` - 薄透鏡公式
- `create_qubit_state` - 量子位元狀態
- `apply_pauli_gates` - Pauli閘操作
- `calculate_commutator` - 對易子計算

### 🌤️ 氣象工具
- `get_forecast` - 天氣預報
- `get_alerts` - 天氣警報
- `create_thumbnail` - 圖像縮略圖

## 💡 使用範例

### 🧮 數學計算
```python
# 簡化表達式
await simplify_expression("x**2 + 2*x + 1")
# 輸出: (x + 1)**2

# 求導數
await differentiate_expression("x**3 + 2*x", "x")
# 輸出: 3*x**2 + 2

# 解方程
await solve_equation("x**2 - 4", "x")
# 輸出: x = -2, 2
```

### 🔐 密碼學
```python
# 凱撒密碼
await encrypt_shift_cipher("HELLO", 3)
# 輸出: KHOOR

# 維吉尼亞密碼
await encrypt_vigenere_cipher("ATTACKATDAWN", "LEMON")
# 輸出: LXFOPVEFRNHR

# RSA加密
await generate_rsa_keys(512)
```

### ⚛️ 物理計算
```python
# 動能計算
await calculate_kinetic_energy(10, 5)
# 輸出: KE = 125 J

# 單位轉換
await convert_units(100, "meter", "m")

# 量子位元
await create_qubit_state(0.707, 0.707)
# 創建 |+⟩ 狀態
```

### 🌤️ 天氣查詢
```python
# 天氣預報
await get_forecast(37.7749, -122.4194)  # 舊金山

# 天氣警報
await get_alerts("CA")  # 加州警報
```

## 🎯 目標使用者

- **👩‍🏫 教育工作者**: 物理、數學、密碼學課程教學
- **👨‍🎓 學生群體**: 學習輔助和作業驗證工具
- **🔬 研究人員**: 快速計算驗證和原型開發
- **💻 開發者**: 算法實現參考和數學函數驗證

## 🌐 技術架構

```
FastMCP Server
├── 依賴套件: ["Pillow", "mcp-server-time", "sympy"]
├── HTTP客戶端: httpx (用於天氣API)
├── 圖像處理: PIL/Pillow
└── 符號計算: SymPy生態系統
    ├── sympy.crypto (密碼學)
    ├── sympy.physics (物理學)
    └── 核心數學功能
```

## ✨ 主要特色

- **🔧 多領域整合**: 單一服務器涵蓋數學、物理、密碼學、氣象
- **🎯 符號計算**: 精確計算，LaTeX格式輸出
- **🌍 中文支援**: 完整的繁體中文界面
- **📚 教育友善**: 詳細說明和錯誤處理
- **🔨 可擴展**: 模組化設計，易於添加新功能

## 🔮 未來發展

1. **擴展物理模組**: 電磁學、熱力學計算
2. **增強密碼學**: 現代密碼算法、區塊鏈相關
3. **數學可視化**: 函數繪圖、3D圖形
4. **機器學習整合**: 科學計算與AI結合
5. **雲端計算**: 大規模數值計算支援

## 📄 授權

MIT License

## 🤝 貢獻

歡迎提交問題和拉取請求來改進這個項目！

---

這個MCP服務器代表了**跨學科計算工具**的創新嘗試，將傳統的單一功能服務器發展為多領域科學計算平台，為STEM教育和研究提供強大的技術支援！
