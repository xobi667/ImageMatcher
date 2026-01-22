# -*- coding: utf-8 -*-
import subprocess
import sys
import os

def main():
    print("=" * 44)
    print("      ImageMatcher - Install Dependencies")
    print("=" * 44)
    print()

    python = sys.executable
    print(f"Python: {python}")
    print()

    # Step 1: 确保 pip
    print("Step 1: Checking pip...")
    result = subprocess.run([python, '-m', 'pip', '--version'],
                          capture_output=True, text=True)

    if result.returncode != 0:
        print("Installing pip...")
        subprocess.run([python, '-m', 'ensurepip', '--upgrade'], check=False)
        print()
        print("pip installed! Please run install.bat AGAIN.")
        print()
        return
    else:
        print(f"pip OK: {result.stdout.strip()}")
    print()

    # Step 2: 安装依赖
    print("Step 2: Installing packages...")
    print()

    subprocess.run([
        python, '-m', 'pip', 'install',
        'flask', 'pillow', 'numpy', 'scipy',
        '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple',
        '--quiet'
    ], check=False)

    print()
    print("=" * 44)
    print("      Done! Run 'run.bat' to start")
    print("=" * 44)

if __name__ == '__main__':
    main()
    input("\nPress Enter to exit...")
