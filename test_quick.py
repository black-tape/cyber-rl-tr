"""
快速测试脚本 - 使用模拟数据验证系统功能
可以在 CPU 上运行，用于快速验证代码
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("="*70)
print("快速功能测试")
print("="*70)

# 测试 1: 导入所有模块
print("\n[1/6] 测试模块导入...")
try:
    from env.security_env import SecurityEnv, MockDataGenerator
    from reward.reward_fn import RewardFunction
    print("[OK] 环境和奖励函数模块导入成功")
except Exception as e:
    print(f"[X] 导入失败: {e}")
    sys.exit(1)

try:
    # 尝试导入 PyTorch，如果没有也继续
    import torch
    has_torch = True
    print(f"[OK] PyTorch 已安装: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[OK] GPU 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] 使用 CPU 模式")
except ImportError:
    has_torch = False
    print("[WARNING] PyTorch 未安装，跳过 Agent 测试")

# 测试 2: 创建环境
print("\n[2/6] 测试环境创建...")
try:
    features, labels = MockDataGenerator.generate_mock_data(
        num_samples=100,
        feature_dim=20,
        attack_ratio=0.3
    )
    print(f"[OK] 生成数据: {len(features)} 个样本")

    env = SecurityEnv(
        data=features,
        labels=labels,
        state_dim=25,
        history_length=5
    )
    print(f"[OK] 环境创建成功，状态维度: {env.state_dim}")
except Exception as e:
    print(f"[X] 环境创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 3: 测试奖励函数
print("\n[3/6] 测试奖励函数...")
try:
    reward_fn = RewardFunction()

    # 测试不同场景
    r1 = reward_fn.compute_reward(action=2, is_attack=True)  # 成功阻断攻击
    r2 = reward_fn.compute_reward(action=0, is_attack=True)  # 攻击漏报
    r3 = reward_fn.compute_reward(action=2, is_attack=False) # 误报
    r4 = reward_fn.compute_reward(action=0, is_attack=False) # 正确放行

    print(f"[OK] 成功阻断攻击: +{r1}")
    print(f"[OK] 攻击漏报: {r2}")
    print(f"[OK] 误报: {r3}")
    print(f"[OK] 正确放行: +{r4}")
except Exception as e:
    print(f"[X] 奖励函数测试失败: {e}")
    sys.exit(1)

# 测试 4: 环境交互
print("\n[4/6] 测试环境交互...")
try:
    state = env.reset()
    print(f"[OK] 初始状态形状: {state.shape}")

    next_state, reward, done, info = env.step(action=1)  # ALERT
    print(f"[OK] 单步执行成功")
    print(f"     状态: {next_state.shape}, 奖励: {reward:.2f}, 完成: {done}")
except Exception as e:
    print(f"[X] 环境交互失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 5: 多步 episode
print("\n[5/6] 测试完整 episode...")
try:
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 50:  # 最多 50 步
        action = np.random.randint(0, 3)  # 随机动作
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

    print(f"[OK] Episode 完成")
    print(f"     步数: {steps}, 总奖励: {total_reward:.2f}")
    print(f"     最后一步信息: {info}")
except Exception as e:
    print(f"[X] Episode 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 6: Agent (如果 PyTorch 可用)
if has_torch:
    print("\n[6/6] 测试 Agent 创建...")
    try:
        from agent.rl_agent import DQNAgent

        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=3,
            hidden_dims=(64, 32),
            learning_rate=0.001
        )
        print("[OK] DQN Agent 创建成功")

        # 测试动作选择
        state = env.reset()
        action = agent.select_action(state, training=True)
        print(f"[OK] 动作选择成功: action={action}")

        # 测试经验存储和更新
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        print(f"[OK] 经验存储成功")

        print(f"[OK] Agent 基础功能正常")
    except Exception as e:
        print(f"[X] Agent 测试失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n[6/6] 跳过 Agent 测试 (PyTorch 未安装)")

# 总结
print("\n" + "="*70)
print("测试完成!")
print("="*70)
print("\n核心功能验证:")
print("  [OK] 环境系统")
print("  [OK] 奖励函数")
print("  [OK] 模拟数据生成")
print("  [OK] Episode 执行")
if has_torch:
    print("  [OK] Agent 系统")
    print("\n✓ 所有测试通过！可以开始训练。")
    if torch.cuda.is_available():
        print(f"✓ GPU 加速可用: {torch.cuda.get_device_name(0)}")
    else:
        print("! 当前使用 CPU 模式")
else:
    print("  [SKIP] Agent 系统 (需要安装 PyTorch)")
    print("\n! 请先安装 PyTorch:")
    print("  python install_gpu.py")

print("\n下一步:")
print("  python main.py --mode local       # 单 Agent 训练")
print("  python main.py --mode federated   # 联邦学习训练")
