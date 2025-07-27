#!/usr/bin/env python3
"""
测试运行脚本 - 提供便捷的测试执行方式
"""
import sys
import os
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description):
    """运行命令并处理结果"""
    print(f"\n{'='*50}")
    print(f"运行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("警告:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: 命令执行失败 (退出码: {e.returncode})")
        print("标准输出:")
        print(e.stdout)
        print("错误输出:")
        print(e.stderr)
        return False


def run_unit_tests():
    """运行单元测试"""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "-m", "not integration and not slow",
        "--tb=short"
    ]
    return run_command(cmd, "单元测试")


def run_integration_tests():
    """运行集成测试"""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "-m", "integration and not slow",
        "--tb=short"
    ]
    return run_command(cmd, "集成测试")


def run_slow_tests():
    """运行慢速测试"""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "-m", "slow",
        "--tb=short"
    ]
    return run_command(cmd, "慢速测试")


def run_all_tests():
    """运行所有测试"""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "--tb=short"
    ]
    return run_command(cmd, "所有测试")


def run_coverage_tests():
    """运行带覆盖率的测试"""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "--cov=app",
        "--cov=main",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--tb=short"
    ]
    return run_command(cmd, "覆盖率测试")


def run_specific_test(test_path):
    """运行特定测试"""
    cmd = [
        sys.executable, "-m", "pytest", 
        test_path, 
        "-v", 
        "--tb=short"
    ]
    return run_command(cmd, f"特定测试: {test_path}")


def run_by_marker(marker):
    """按标记运行测试"""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "-m", marker,
        "--tb=short"
    ]
    return run_command(cmd, f"标记测试: {marker}")


def validate_environment():
    """验证测试环境"""
    print("验证测试环境...")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查必要的包
    required_packages = ['pytest', 'pytest-cov']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")
    
    if missing_packages:
        print(f"\n请安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # 检查项目结构
    required_dirs = ['app', 'tests']
    for dir_name in required_dirs:
        if (project_root / dir_name).exists():
            print(f"✓ {dir_name}/ 目录存在")
        else:
            print(f"✗ {dir_name}/ 目录不存在")
            return False
    
    print("✓ 环境验证通过")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="自动标注项目测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_tests.py --unit                    # 运行单元测试
  python run_tests.py --integration             # 运行集成测试
  python run_tests.py --all                     # 运行所有测试
  python run_tests.py --coverage                # 运行覆盖率测试
  python run_tests.py --specific tests/test_config.py  # 运行特定测试
  python run_tests.py --marker validation       # 运行特定标记的测试
        """
    )
    
    # 测试类型选项
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--unit', action='store_true', help='运行单元测试')
    group.add_argument('--integration', action='store_true', help='运行集成测试')
    group.add_argument('--slow', action='store_true', help='运行慢速测试')
    group.add_argument('--all', action='store_true', help='运行所有测试')
    group.add_argument('--coverage', action='store_true', help='运行带覆盖率的测试')
    group.add_argument('--specific', type=str, help='运行特定测试文件或函数')
    group.add_argument('--marker', type=str, help='按标记运行测试')
    group.add_argument('--validate', action='store_true', help='验证测试环境')
    
    args = parser.parse_args()
    
    # 验证环境
    if not validate_environment():
        print("环境验证失败，退出")
        return 1
    
    # 根据参数运行相应的测试
    success = False
    
    if args.validate:
        print("环境验证完成")
        return 0
    elif args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.slow:
        success = run_slow_tests()
    elif args.all:
        success = run_all_tests()
    elif args.coverage:
        success = run_coverage_tests()
    elif args.specific:
        success = run_specific_test(args.specific)
    elif args.marker:
        success = run_by_marker(args.marker)
    
    # 输出结果
    if success:
        print(f"\n{'='*50}")
        print("✓ 测试执行成功！")
        print(f"{'='*50}")
        return 0
    else:
        print(f"\n{'='*50}")
        print("✗ 测试执行失败！")
        print(f"{'='*50}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 