from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.prompts import base
from PIL import Image as PILImage
import sympy as sp
from sympy import symbols, sympify, latex, pi, E, I, sqrt, cos, sin, exp
from sympy.crypto.crypto import (
    encipher_shift, decipher_shift,
    encipher_vigenere, decipher_vigenere,
    encipher_affine, decipher_affine,
    encipher_substitution,
    encipher_hill, decipher_hill,
    encipher_railfence, decipher_railfence,
    rsa_public_key, rsa_private_key, encipher_rsa, decipher_rsa,
    encode_morse, decode_morse,
    encipher_bifid, decipher_bifid
)
from sympy.physics.units import (
    meter, kilogram, second, ampere, kelvin, mole, candela,
    newton, joule, watt, pascal, hertz, coulomb, volt, ohm,
    speed_of_light, planck, boltzmann, avogadro,
    convert_to
)
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, Particle
from sympy.physics.optics import RayTransferMatrix, ThinLens, FreeSpace
from sympy.physics.quantum import (
    Qubit, measure_all, qapply, Ket, Bra,
    Commutator, Dagger, TensorProduct
)
from sympy.physics.quantum.pauli import SigmaX, SigmaY, SigmaZ
from sympy.physics.quantum.spin import JxKet, Jz

# Initialize FastMCP server
mcp = FastMCP("weather", dependencies=["Pillow","mcp-server-time", "sympy"])

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""

@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)

@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]

@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """Create a thumbnail from an image"""
    img = PILImage.open(image_path)
    img.thumbnail((100, 100))
    return Image(data=img.tobytes(), format="png")

@mcp.tool()
async def simplify_expression(expression: str) -> str:
    """簡化數學表達式
    
    Args:
        expression: 要簡化的數學表達式（例如：'x**2 + 2*x + 1'）
    """
    try:
        expr = sympify(expression)
        simplified = sp.simplify(expr)
        return f"原始表達式: {expression}\n簡化結果: {simplified}\nLaTeX格式: {latex(simplified)}"
    except Exception as e:
        return f"無法簡化表達式：{str(e)}"

@mcp.tool()
async def differentiate_expression(expression: str, variable: str = "x") -> str:
    """對表達式求導
    
    Args:
        expression: 要求導的數學表達式
        variable: 求導變量（預設為'x'）
    """
    try:
        expr = sympify(expression)
        var = symbols(variable)
        derivative = sp.diff(expr, var)
        return f"原始函數: {expression}\n對{variable}求導: {derivative}\nLaTeX格式: {latex(derivative)}"
    except Exception as e:
        return f"無法計算導數：{str(e)}"

@mcp.tool()
async def integrate_expression(expression: str, variable: str = "x", definite: bool = False, lower_limit: str = "", upper_limit: str = "") -> str:
    """對表達式求積分
    
    Args:
        expression: 要積分的數學表達式
        variable: 積分變量（預設為'x'）
        definite: 是否為定積分
        lower_limit: 定積分下限（如果definite為True）
        upper_limit: 定積分上限（如果definite為True）
    """
    try:
        expr = sympify(expression)
        var = symbols(variable)
        
        if definite and lower_limit and upper_limit:
            lower = sympify(lower_limit)
            upper = sympify(upper_limit)
            integral = sp.integrate(expr, (var, lower, upper))
            return f"原始函數: {expression}\n定積分[{lower_limit}, {upper_limit}]: {integral}\nLaTeX格式: {latex(integral)}"
        else:
            integral = sp.integrate(expr, var)
            return f"原始函數: {expression}\n不定積分: {integral} + C\nLaTeX格式: {latex(integral)} + C"
    except Exception as e:
        return f"無法計算積分：{str(e)}"

@mcp.tool()
async def solve_equation(equation: str, variable: str = "x") -> str:
    """解方程
    
    Args:
        equation: 要解的方程（例如：'x**2 - 4'）
        variable: 要解的變量（預設為'x'）
    """
    try:
        expr = sympify(equation)
        var = symbols(variable)
        solutions = sp.solve(expr, var)
        
        if not solutions:
            return f"方程 {equation} = 0 無解"
        
        solutions_str = [str(sol) for sol in solutions]
        return f"方程: {equation} = 0\n解: {variable} = {', '.join(solutions_str)}"
    except Exception as e:
        return f"無法解方程：{str(e)}"

@mcp.tool()
async def calculate_limit(expression: str, variable: str = "x", approach_value: str = "0") -> str:
    """計算極限
    
    Args:
        expression: 要計算極限的表達式
        variable: 變量（預設為'x'）
        approach_value: 趨近值（預設為'0'）
    """
    try:
        expr = sympify(expression)
        var = symbols(variable)
        approach = sympify(approach_value)
        
        limit_result = sp.limit(expr, var, approach)
        return f"極限: lim({variable} → {approach_value}) {expression} = {limit_result}\nLaTeX格式: {latex(limit_result)}"
    except Exception as e:
        return f"無法計算極限：{str(e)}"

@mcp.tool()
async def factor_expression(expression: str) -> str:
    """因式分解
    
    Args:
        expression: 要因式分解的表達式
    """
    try:
        expr = sympify(expression)
        factored = sp.factor(expr)
        return f"原始表達式: {expression}\n因式分解: {factored}\nLaTeX格式: {latex(factored)}"
    except Exception as e:
        return f"無法因式分解：{str(e)}"

@mcp.tool()
async def expand_expression(expression: str) -> str:
    """展開表達式
    
    Args:
        expression: 要展開的表達式（例如：'(x+1)**2'）
    """
    try:
        expr = sympify(expression)
        expanded = sp.expand(expr)
        return f"原始表達式: {expression}\n展開結果: {expanded}\nLaTeX格式: {latex(expanded)}"
    except Exception as e:
        return f"無法展開表達式：{str(e)}"

@mcp.tool()
async def encrypt_shift_cipher(message: str, shift: int) -> str:
    """使用移位密碼（凱撒密碼）加密訊息
    
    Args:
        message: 要加密的訊息
        shift: 移位數量（0-25）
    """
    try:
        encrypted = encipher_shift(message.upper(), shift)
        return f"原始訊息: {message}\n移位數: {shift}\n加密結果: {encrypted}"
    except Exception as e:
        return f"加密失敗：{str(e)}"

@mcp.tool()
async def decrypt_shift_cipher(ciphertext: str, shift: int) -> str:
    """使用移位密碼（凱撒密碼）解密訊息
    
    Args:
        ciphertext: 要解密的密文
        shift: 移位數量（0-25）
    """
    try:
        decrypted = decipher_shift(ciphertext.upper(), shift)
        return f"密文: {ciphertext}\n移位數: {shift}\n解密結果: {decrypted}"
    except Exception as e:
        return f"解密失敗：{str(e)}"

@mcp.tool()
async def encrypt_vigenere_cipher(message: str, keyword: str) -> str:
    """使用維吉尼亞密碼加密訊息
    
    Args:
        message: 要加密的訊息
        keyword: 密碼關鍵字
    """
    try:
        encrypted = encipher_vigenere(message.upper(), keyword.upper())
        return f"原始訊息: {message}\n關鍵字: {keyword}\n加密結果: {encrypted}"
    except Exception as e:
        return f"加密失敗：{str(e)}"

@mcp.tool()
async def decrypt_vigenere_cipher(ciphertext: str, keyword: str) -> str:
    """使用維吉尼亞密碼解密訊息
    
    Args:
        ciphertext: 要解密的密文
        keyword: 密碼關鍵字
    """
    try:
        decrypted = decipher_vigenere(ciphertext.upper(), keyword.upper())
        return f"密文: {ciphertext}\n關鍵字: {keyword}\n解密結果: {decrypted}"
    except Exception as e:
        return f"解密失敗：{str(e)}"

@mcp.tool()
async def encrypt_affine_cipher(message: str, a: int, b: int) -> str:
    """使用仿射密碼加密訊息
    
    Args:
        message: 要加密的訊息
        a: 乘法密鑰（必須與26互質）
        b: 加法密鑰（0-25）
    """
    try:
        encrypted = encipher_affine(message.upper(), (a, b))
        return f"原始訊息: {message}\n密鑰 (a,b): ({a},{b})\n加密結果: {encrypted}"
    except Exception as e:
        return f"加密失敗：{str(e)}"

@mcp.tool()
async def decrypt_affine_cipher(ciphertext: str, a: int, b: int) -> str:
    """使用仿射密碼解密訊息
    
    Args:
        ciphertext: 要解密的密文
        a: 乘法密鑰（必須與26互質）
        b: 加法密鑰（0-25）
    """
    try:
        decrypted = decipher_affine(ciphertext.upper(), (a, b))
        return f"密文: {ciphertext}\n密鑰 (a,b): ({a},{b})\n解密結果: {decrypted}"
    except Exception as e:
        return f"解密失敗：{str(e)}"

@mcp.tool()
async def encrypt_railfence_cipher(message: str, rails: int) -> str:
    """使用柵欄密碼加密訊息
    
    Args:
        message: 要加密的訊息
        rails: 柵欄數量
    """
    try:
        encrypted = encipher_railfence(message.lower(), rails)
        return f"原始訊息: {message}\n柵欄數: {rails}\n加密結果: {encrypted}"
    except Exception as e:
        return f"加密失敗：{str(e)}"

@mcp.tool()
async def decrypt_railfence_cipher(ciphertext: str, rails: int) -> str:
    """使用柵欄密碼解密訊息
    
    Args:
        ciphertext: 要解密的密文
        rails: 柵欄數量
    """
    try:
        decrypted = decipher_railfence(ciphertext.lower(), rails)
        return f"密文: {ciphertext}\n柵欄數: {rails}\n解密結果: {decrypted}"
    except Exception as e:
        return f"解密失敗：{str(e)}"

@mcp.tool()
async def generate_rsa_keys(bits: int = 1024) -> str:
    """生成RSA公私鑰對
    
    Args:
        bits: 密鑰長度（預設1024位）
    """
    try:
        private_key = rsa_private_key(bits)
        public_key = rsa_public_key(private_key)
        
        return f"RSA密鑰對生成成功：\n私鑰: {private_key}\n公鑰: {public_key}\n密鑰長度: {bits}位"
    except Exception as e:
        return f"密鑰生成失敗：{str(e)}"

@mcp.tool()
async def encrypt_rsa(message: int, public_key_n: int, public_key_e: int) -> str:
    """使用RSA公鑰加密數字
    
    Args:
        message: 要加密的數字
        public_key_n: RSA公鑰的n值
        public_key_e: RSA公鑰的e值
    """
    try:
        encrypted = encipher_rsa(message, (public_key_n, public_key_e))
        return f"原始訊息: {message}\n公鑰 (n,e): ({public_key_n},{public_key_e})\n加密結果: {encrypted}"
    except Exception as e:
        return f"RSA加密失敗：{str(e)}"

@mcp.tool()
async def decrypt_rsa(ciphertext: int, private_key_p: int, private_key_q: int, private_key_d: int) -> str:
    """使用RSA私鑰解密數字
    
    Args:
        ciphertext: 要解密的密文
        private_key_p: RSA私鑰的p值
        private_key_q: RSA私鑰的q值
        private_key_d: RSA私鑰的d值
    """
    try:
        decrypted = decipher_rsa(ciphertext, (private_key_p, private_key_q, private_key_d))
        return f"密文: {ciphertext}\n私鑰 (p,q,d): ({private_key_p},{private_key_q},{private_key_d})\n解密結果: {decrypted}"
    except Exception as e:
        return f"RSA解密失敗：{str(e)}"

@mcp.tool()
async def encode_morse_code(message: str) -> str:
    """將文字轉換為摩斯密碼
    
    Args:
        message: 要編碼的文字訊息
    """
    try:
        morse = encode_morse(message.upper())
        return f"原始訊息: {message}\n摩斯密碼: {morse}"
    except Exception as e:
        return f"摩斯密碼編碼失敗：{str(e)}"

@mcp.tool()
async def decode_morse_code(morse_code: str) -> str:
    """將摩斯密碼轉換為文字
    
    Args:
        morse_code: 要解碼的摩斯密碼（用空格分隔字母，用/分隔單詞）
    """
    try:
        decoded = decode_morse(morse_code)
        return f"摩斯密碼: {morse_code}\n解碼結果: {decoded}"
    except Exception as e:
        return f"摩斯密碼解碼失敗：{str(e)}"

@mcp.tool()
async def encrypt_substitution_cipher(message: str, key: str) -> str:
    """使用替換密碼加密訊息
    
    Args:
        message: 要加密的訊息
        key: 26個字母的替換密鑰（例如：'ZYXWVUTSRQPONMLKJIHGFEDCBA'）
    """
    try:
        if len(key) != 26:
            return "替換密鑰必須包含26個字母"
        
        encrypted = encipher_substitution(message.upper(), key.upper())
        return f"原始訊息: {message}\n替換密鑰: {key}\n加密結果: {encrypted}"
    except Exception as e:
        return f"替換密碼加密失敗：{str(e)}"

@mcp.tool()
async def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """單位轉換工具
    
    Args:
        value: 數值
        from_unit: 原始單位（如：meter, kilogram, second等）
        to_unit: 目標單位
    """
    try:
        # 定義常用單位對應表
        unit_map = {
            'meter': meter, 'm': meter,
            'kilogram': kilogram, 'kg': kilogram,
            'second': second, 's': second,
            'newton': newton, 'N': newton,
            'joule': joule, 'J': joule,
            'watt': watt, 'W': watt,
            'pascal': pascal, 'Pa': pascal,
            'hertz': hertz, 'Hz': hertz,
            'coulomb': coulomb, 'C': coulomb,
            'volt': volt, 'V': volt,
            'ohm': ohm, 'Ω': ohm
        }
        
        from_u = unit_map.get(from_unit)
        to_u = unit_map.get(to_unit)
        
        if not from_u or not to_u:
            return f"不支援的單位：{from_unit} 或 {to_unit}"
        
        quantity = value * from_u
        converted = convert_to(quantity, to_u)
        
        return f"單位轉換結果：\n{value} {from_unit} = {float(converted/to_u)} {to_unit}"
    except Exception as e:
        return f"單位轉換失敗：{str(e)}"

@mcp.tool()
async def calculate_kinetic_energy(mass: float, velocity: float) -> str:
    """計算動能 KE = 1/2 * m * v²
    
    Args:
        mass: 質量（公斤）
        velocity: 速度（公尺/秒）
    """
    try:
        ke = sp.Rational(1, 2) * mass * velocity**2
        return f"動能計算：\n質量: {mass} kg\n速度: {velocity} m/s\n動能: {ke} J = {float(ke)} J"
    except Exception as e:
        return f"動能計算失敗：{str(e)}"

@mcp.tool()
async def calculate_potential_energy(mass: float, height: float, gravity: float = 9.81) -> str:
    """計算重力位能 PE = m * g * h
    
    Args:
        mass: 質量（公斤）
        height: 高度（公尺）
        gravity: 重力加速度（預設9.81 m/s²）
    """
    try:
        pe = mass * gravity * height
        return f"重力位能計算：\n質量: {mass} kg\n高度: {height} m\n重力加速度: {gravity} m/s²\n位能: {pe} J"
    except Exception as e:
        return f"位能計算失敗：{str(e)}"

@mcp.tool()
async def calculate_force(mass: float, acceleration: float) -> str:
    """計算力 F = m * a（牛頓第二定律）
    
    Args:
        mass: 質量（公斤）
        acceleration: 加速度（公尺/秒²）
    """
    try:
        force = mass * acceleration
        return f"力的計算（牛頓第二定律）：\n質量: {mass} kg\n加速度: {acceleration} m/s²\n力: {force} N"
    except Exception as e:
        return f"力計算失敗：{str(e)}"

@mcp.tool()
async def calculate_momentum(mass: float, velocity: float) -> str:
    """計算動量 p = m * v
    
    Args:
        mass: 質量（公斤）
        velocity: 速度（公尺/秒）
    """
    try:
        momentum = mass * velocity
        return f"動量計算：\n質量: {mass} kg\n速度: {velocity} m/s\n動量: {momentum} kg⋅m/s"
    except Exception as e:
        return f"動量計算失敗：{str(e)}"

@mcp.tool()
async def calculate_wave_properties(frequency: float = None, wavelength: float = None, wave_speed: float = None) -> str:
    """計算波的性質 v = f * λ
    
    Args:
        frequency: 頻率（赫茲），提供其中兩個參數
        wavelength: 波長（公尺）
        wave_speed: 波速（公尺/秒，光速約3×10⁸）
    """
    try:
        provided = sum([x is not None for x in [frequency, wavelength, wave_speed]])
        if provided != 2:
            return "請提供三個參數中的任意兩個（頻率、波長、波速）"
        
        if frequency is None:
            frequency = wave_speed / wavelength
            missing = "頻率"
            result = frequency
            unit = "Hz"
        elif wavelength is None:
            wavelength = wave_speed / frequency
            missing = "波長"
            result = wavelength
            unit = "m"
        else:  # wave_speed is None
            wave_speed = frequency * wavelength
            missing = "波速"
            result = wave_speed
            unit = "m/s"
        
        return f"波的性質計算：\n頻率: {frequency} Hz\n波長: {wavelength} m\n波速: {wave_speed} m/s\n計算得出的{missing}: {result} {unit}"
    except Exception as e:
        return f"波性質計算失敗：{str(e)}"

@mcp.tool()
async def calculate_photon_energy(frequency: float = None, wavelength: float = None) -> str:
    """計算光子能量 E = h * f = h * c / λ
    
    Args:
        frequency: 頻率（赫茲）
        wavelength: 波長（公尺）
    """
    try:
        if frequency is None and wavelength is None:
            return "請提供頻率或波長"
        
        h = planck  # 普朗克常數
        c = speed_of_light  # 光速
        
        if frequency is not None:
            energy = h * frequency
            return f"光子能量計算：\n頻率: {frequency} Hz\n能量: {energy} = {float(energy)} J"
        else:
            energy = h * c / wavelength
            return f"光子能量計算：\n波長: {wavelength} m\n能量: {energy} = {float(energy)} J"
    except Exception as e:
        return f"光子能量計算失敗：{str(e)}"

@mcp.tool()
async def thin_lens_calculation(focal_length: float, object_distance: float) -> str:
    """薄透鏡公式計算 1/f = 1/u + 1/v
    
    Args:
        focal_length: 焦距（公尺）
        object_distance: 物距（公尺）
    """
    try:
        # 1/f = 1/u + 1/v => 1/v = 1/f - 1/u
        image_distance = 1 / (1/focal_length - 1/object_distance)
        magnification = -image_distance / object_distance
        
        return f"薄透鏡計算：\n焦距: {focal_length} m\n物距: {object_distance} m\n像距: {image_distance:.4f} m\n放大率: {magnification:.4f}"
    except Exception as e:
        return f"薄透鏡計算失敗：{str(e)}"

@mcp.tool()
async def create_qubit_state(alpha: float, beta: float) -> str:
    """創建量子位元狀態 |ψ⟩ = α|0⟩ + β|1⟩
    
    Args:
        alpha: |0⟩狀態的振幅
        beta: |1⟩狀態的振幅
    """
    try:
        # 檢查歸一化條件
        norm_squared = alpha**2 + beta**2
        if abs(norm_squared - 1) > 0.001:
            return f"警告：狀態未歸一化，|α|² + |β|² = {norm_squared:.4f} ≠ 1"
        
        # 創建量子位元狀態
        state = alpha * Qubit('0') + beta * Qubit('1')
        
        return f"量子位元狀態：\n|ψ⟩ = {alpha}|0⟩ + {beta}|1⟩\n狀態表示: {state}\n歸一化檢查: |α|² + |β|² = {norm_squared:.4f}"
    except Exception as e:
        return f"量子位元狀態創建失敗：{str(e)}"

@mcp.tool()
async def apply_pauli_gates(gate_type: str, qubit_state: str = "0") -> str:
    """應用Pauli閘到量子位元
    
    Args:
        gate_type: 閘類型（X, Y, Z）
        qubit_state: 初始量子位元狀態（'0'或'1'）
    """
    try:
        # 選擇Pauli閘
        if gate_type.upper() == 'X':
            gate = SigmaX()
        elif gate_type.upper() == 'Y':
            gate = SigmaY()
        elif gate_type.upper() == 'Z':
            gate = SigmaZ()
        else:
            return "不支援的閘類型，請使用 X, Y, 或 Z"
        
        # 創建初始狀態
        initial_state = Qubit(qubit_state)
        
        # 應用閘
        result = qapply(gate * initial_state)
        
        return f"Pauli-{gate_type.upper()}閘應用：\n初始狀態: |{qubit_state}⟩\n應用 {gate_type.upper()} 閘\n結果狀態: {result}"
    except Exception as e:
        return f"Pauli閘應用失敗：{str(e)}"

@mcp.tool()
async def calculate_commutator(operator_a: str, operator_b: str) -> str:
    """計算量子算符的對易子 [A, B] = AB - BA
    
    Args:
        operator_a: 第一個算符（如：'X', 'Y', 'Z'）
        operator_b: 第二個算符
    """
    try:
        # 簡化的Pauli算符對易子計算
        pauli_map = {'X': SigmaX(), 'Y': SigmaY(), 'Z': SigmaZ()}
        
        if operator_a not in pauli_map or operator_b not in pauli_map:
            return "目前只支援Pauli算符 X, Y, Z"
        
        A = pauli_map[operator_a]
        B = pauli_map[operator_b]
        
        comm = Commutator(A, B)
        result = comm.doit()
        
        return f"對易子計算：\n[{operator_a}, {operator_b}] = {operator_a}{operator_b} - {operator_b}{operator_a}\n結果: {result}"
    except Exception as e:
        return f"對易子計算失敗：{str(e)}"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')