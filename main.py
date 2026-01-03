"""
多 Agent 强化学习 + 强化型联邦学习安全防御系统
主入口文件
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from train.train_local import train_local
from train.train_federated import train_federated


def main():
    parser = argparse.ArgumentParser(
        description='多 Agent 强化学习 + 强化型联邦学习安全防御系统'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['local', 'federated'],
        default='federated',
        help='训练模式: local (单 Agent 本地训练) 或 federated (多 Agent 联邦训练)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='配置文件路径'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("多 Agent 强化学习 + 强化型联邦学习安全防御系统")
    print("=" * 70)
    print(f"训练模式: {args.mode.upper()}")
    print(f"配置文件: {args.config}")
    print("=" * 70 + "\n")

    if args.mode == 'local':
        print("启动本地训练...")
        train_local(args.config)
    elif args.mode == 'federated':
        print("启动联邦训练...")
        train_federated(args.config)
    else:
        print(f"未知的训练模式: {args.mode}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
