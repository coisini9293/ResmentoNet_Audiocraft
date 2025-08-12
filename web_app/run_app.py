"""
启动情绪音乐助手Web应用
用法：
  python web_app/run_app.py                 # 本机访问 http://127.0.0.1:8501
  python web_app/run_app.py --public        # 本机监听 0.0.0.0，局域网访问
  python web_app/run_app.py --tunnel        # 使用 Cloudflare Tunnel 暴露临时公网 URL
"""

import os
import sys
import subprocess
import argparse

def install_requirements():
    """安装依赖包"""
    print("📦 正在安装依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖包安装完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False
    return True

def run_streamlit(public: bool = False):
    """运行Streamlit应用"""
    print("🚀 启动情绪音乐助手...")
    try:
        # 设置环境变量（UTF-8 与 ASCII 安全的临时/结果目录，避免 Windows 下中文路径导致的 ascii 编码错误）
        os.environ.setdefault('PYTHONUTF8', '1')
        os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

        # 选择 ASCII 安全的临时目录与结果目录
        ascii_tmp = os.environ.get('ASCII_TMP') or (r"C:\\Windows\\Temp")
        try:
            os.makedirs(ascii_tmp, exist_ok=True)
        except Exception:
            pass
        for k in ['TMP', 'TEMP', 'TMPDIR']:
            os.environ.setdefault(k, ascii_tmp)

        # 结果目录优先使用环境变量 RESULT_DIR；若未设置，则放到 ASCII 安全路径
        if not os.environ.get('RESULT_DIR'):
            default_result = os.path.join(ascii_tmp, 'emotion_music_result')
            try:
                os.makedirs(default_result, exist_ok=True)
                os.environ['RESULT_DIR'] = default_result
            except Exception:
                # 兜底为当前目录下 result（可能包含中文，但至少可用）
                os.environ.setdefault('RESULT_DIR', 'result')

        # Streamlit 网络配置
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0' if public else '127.0.0.1'
        
        # 启动Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0" if public else "127.0.0.1"
        ])
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")


def run_cloudflare_tunnel(port: int = 8501):
    """使用 cloudflared 建立临时公网隧道（需本机已安装 cloudflared）"""
    print("🌐 尝试启动 Cloudflare Tunnel ...")
    try:
        # 先尝试查找 cloudflared
        which = subprocess.run(["where", "cloudflared"], capture_output=True, text=True, shell=True)
        if which.returncode != 0:
            print("❌ 未找到 cloudflared。请先安装：choco install cloudflared 或到官方发布页下载。")
            return
        # 前台运行隧道（用户可复制生成的 URL）
        subprocess.Popen(["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"])
        print("✅ Cloudflare Tunnel 已启动，终端中会显示公网访问 URL。")
    except Exception as e:
        print(f"❌ 启动 Cloudflare Tunnel 失败: {e}")

def main():
    """主函数"""
    print("🎭 情绪音乐助手 - Web应用")
    print("=" * 50)
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", action="store_true", help="监听 0.0.0.0，局域网可访问")
    parser.add_argument("--tunnel", action="store_true", help="启动 Cloudflare Tunnel 暴露公网 URL")
    args = parser.parse_args()
    
    # 检查当前目录
    if not os.path.exists("main.py"):
        print("❌ 请在web_app目录下运行此脚本")
        return
    
    # 安装依赖
    if not install_requirements():
        return
    
    # 运行应用
    if args.tunnel:
        run_cloudflare_tunnel(port=8501)
        run_streamlit(public=False)
    else:
        run_streamlit(public=args.public)

if __name__ == "__main__":
    main() 