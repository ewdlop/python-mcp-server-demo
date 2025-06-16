from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.prompts import base
from PIL import Image as PILImage
import sympy as sp
from sympy import symbols, sympify, latex
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

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')