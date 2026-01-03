"""
数据集下载和预处理模块
支持 CICIDS2017 和 NSL-KDD 数据集
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import urllib.request
import zipfile


class DatasetDownloader:
    """数据集下载器"""

    # NSL-KDD 数据集 URL
    NSL_KDD_URLS = {
        'train': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt',
        'test': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt'
    }

    # CICIDS2017 数据集说明
    CICIDS2017_INFO = """
    CICIDS2017 数据集下载说明:

    由于 CICIDS2017 数据集较大(约 7GB),需要手动下载:

    1. 访问官方网站: https://www.unb.ca/cic/datasets/ids-2017.html
    2. 下载 CSV 文件
    3. 将文件放置到 data/raw/cicids2017/ 目录下

    或者使用 Kaggle 版本:
    - https://www.kaggle.com/datasets/cicdataset/cicids2017
    """

    @staticmethod
    def download_nsl_kdd(save_dir: str = 'data/raw/nsl_kdd'):
        """
        下载 NSL-KDD 数据集

        Args:
            save_dir: 保存目录
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print("正在下载 NSL-KDD 数据集...")

        for name, url in DatasetDownloader.NSL_KDD_URLS.items():
            file_path = save_path / f'{name}.txt'

            if file_path.exists():
                print(f"{name}.txt 已存在，跳过下载")
                continue

            try:
                print(f"下载 {name}.txt...")
                urllib.request.urlretrieve(url, file_path)
                print(f"✓ {name}.txt 下载完成")
            except Exception as e:
                print(f"✗ 下载 {name}.txt 失败: {e}")
                raise

        print("NSL-KDD 数据集下载完成!")
        return save_path

    @staticmethod
    def check_cicids2017(data_dir: str = 'data/raw/cicids2017'):
        """
        检查 CICIDS2017 数据集是否存在

        Args:
            data_dir: 数据目录
        """
        data_path = Path(data_dir)

        if not data_path.exists():
            print(DatasetDownloader.CICIDS2017_INFO)
            return False

        csv_files = list(data_path.glob('*.csv'))
        if len(csv_files) == 0:
            print(DatasetDownloader.CICIDS2017_INFO)
            return False

        print(f"找到 {len(csv_files)} 个 CICIDS2017 CSV 文件")
        return True


class NSLKDDProcessor:
    """NSL-KDD 数据集处理器"""

    # NSL-KDD 特征名称
    COLUMN_NAMES = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]

    def __init__(self, data_dir: str = 'data/raw/nsl_kdd'):
        """
        初始化处理器

        Args:
            data_dir: 数据目录
        """
        self.data_dir = Path(data_dir)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载数据

        Returns:
            (train_df, test_df)
        """
        train_path = self.data_dir / 'train.txt'
        test_path = self.data_dir / 'test.txt'

        print("加载 NSL-KDD 数据...")

        train_df = pd.read_csv(train_path, names=self.COLUMN_NAMES, header=None)
        test_df = pd.read_csv(test_path, names=self.COLUMN_NAMES, header=None)

        print(f"训练集: {len(train_df)} 条记录")
        print(f"测试集: {len(test_df)} 条记录")

        return train_df, test_df

    def preprocess(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        预处理数据

        Args:
            train_df: 训练数据框
            test_df: 测试数据框

        Returns:
            (X_train, y_train, X_test, y_test)
        """
        print("预处理数据...")

        # 移除 difficulty 列
        train_df = train_df.drop('difficulty', axis=1)
        test_df = test_df.drop('difficulty', axis=1)

        # 提取标签
        y_train = train_df['label'].copy()
        y_test = test_df['label'].copy()

        # 二分类: normal=0, attack=1
        y_train = (y_train != 'normal').astype(int)
        y_test = (y_test != 'normal').astype(int)

        # 移除标签列
        X_train = train_df.drop('label', axis=1)
        X_test = test_df.drop('label', axis=1)

        # 处理类别特征
        categorical_cols = ['protocol_type', 'service', 'flag']

        for col in categorical_cols:
            le = LabelEncoder()
            # 合并训练和测试数据以确保一致的编码
            combined = pd.concat([X_train[col], X_test[col]], axis=0)
            le.fit(combined)

            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])

        # 转换为 numpy 数组
        X_train = X_train.values.astype(np.float32)
        X_test = X_test.values.astype(np.float32)

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = y_train.values.astype(np.int32)
        y_test = y_test.values.astype(np.int32)

        print(f"特征维度: {X_train.shape[1]}")
        print(f"训练集攻击比例: {np.mean(y_train):.2%}")
        print(f"测试集攻击比例: {np.mean(y_test):.2%}")

        return X_train, y_train, X_test, y_test

    def create_non_iid_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_clients: int = 3
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        创建 Non-IID 数据分割

        Args:
            X: 特征数据
            y: 标签
            num_clients: 客户端数量

        Returns:
            (datasets, labels_list)
        """
        print(f"\n创建 Non-IID 数据分割 ({num_clients} 个客户端)...")

        datasets = []
        labels_list = []

        # 分离正常和攻击样本
        normal_indices = np.where(y == 0)[0]
        attack_indices = np.where(y == 1)[0]

        # 为每个客户端分配不同比例的攻击样本
        attack_ratios = np.linspace(0.2, 0.6, num_clients)

        samples_per_client = len(X) // num_clients

        for i in range(num_clients):
            # 计算该客户端的攻击样本数量
            num_attacks = int(samples_per_client * attack_ratios[i])
            num_normal = samples_per_client - num_attacks

            # 随机采样
            client_attack_idx = np.random.choice(attack_indices, num_attacks, replace=False)
            client_normal_idx = np.random.choice(normal_indices, num_normal, replace=False)

            # 合并索引
            client_idx = np.concatenate([client_attack_idx, client_normal_idx])
            np.random.shuffle(client_idx)

            # 提取数据
            client_X = X[client_idx]
            client_y = y[client_idx]

            datasets.append(client_X)
            labels_list.append(client_y)

            print(f"客户端 {i}: {len(client_X)} 个样本, 攻击比例: {np.mean(client_y):.2%}")

        return datasets, labels_list


class CICIDS2017Processor:
    """CICIDS2017 数据集处理器"""

    def __init__(self, data_dir: str = 'data/raw/cicids2017'):
        """
        初始化处理器

        Args:
            data_dir: 数据目录
        """
        self.data_dir = Path(data_dir)

    def load_data(self) -> pd.DataFrame:
        """加载 CICIDS2017 数据"""
        csv_files = list(self.data_dir.glob('*.csv'))

        if len(csv_files) == 0:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到 CSV 文件")

        print(f"加载 {len(csv_files)} 个 CSV 文件...")

        dfs = []
        for csv_file in csv_files:
            print(f"读取 {csv_file.name}...")
            df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
            dfs.append(df)

        # 合并所有数据
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"总共加载 {len(combined_df)} 条记录")

        return combined_df

    def preprocess(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        预处理 CICIDS2017 数据

        Args:
            df: 数据框
            test_size: 测试集比例

        Returns:
            (X_train, y_train, X_test, y_test)
        """
        print("预处理 CICIDS2017 数据...")

        # 标签列通常是最后一列
        label_col = df.columns[-1]

        # 提取标签
        y = (df[label_col] != 'BENIGN').astype(int)

        # 移除标签和无用列
        X = df.drop([label_col], axis=1)

        # 移除非数值列
        X = X.select_dtypes(include=[np.number])

        # 处理缺失值和无穷值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        # 转换为 numpy 数组
        X = X.values.astype(np.float32)
        y = y.values.astype(np.int32)

        # 标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"特征维度: {X_train.shape[1]}")
        print(f"训练集: {len(X_train)} 条, 攻击比例: {np.mean(y_train):.2%}")
        print(f"测试集: {len(X_test)} 条, 攻击比例: {np.mean(y_test):.2%}")

        return X_train, y_train, X_test, y_test


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='数据集下载和预处理')
    parser.add_argument('--dataset', type=str, choices=['nsl_kdd', 'cicids2017'],
                        default='nsl_kdd', help='数据集名称')
    parser.add_argument('--download', action='store_true', help='下载数据集')
    parser.add_argument('--preprocess', action='store_true', help='预处理数据集')
    parser.add_argument('--num_clients', type=int, default=3, help='客户端数量')

    args = parser.parse_args()

    if args.dataset == 'nsl_kdd':
        if args.download:
            DatasetDownloader.download_nsl_kdd()

        if args.preprocess:
            processor = NSLKDDProcessor()
            train_df, test_df = processor.load_data()
            X_train, y_train, X_test, y_test = processor.preprocess(train_df, test_df)

            # 创建 Non-IID 分割
            datasets, labels_list = processor.create_non_iid_splits(
                X_train, y_train, args.num_clients
            )

            # 保存处理后的数据
            save_dir = Path('data/processed/nsl_kdd')
            save_dir.mkdir(parents=True, exist_ok=True)

            np.save(save_dir / 'X_test.npy', X_test)
            np.save(save_dir / 'y_test.npy', y_test)

            for i, (X, y) in enumerate(zip(datasets, labels_list)):
                np.save(save_dir / f'client_{i}_X.npy', X)
                np.save(save_dir / f'client_{i}_y.npy', y)

            print(f"\n数据已保存至 {save_dir}")

    elif args.dataset == 'cicids2017':
        if not DatasetDownloader.check_cicids2017():
            print("请先手动下载 CICIDS2017 数据集")
            return

        if args.preprocess:
            processor = CICIDS2017Processor()
            df = processor.load_data()
            X_train, y_train, X_test, y_test = processor.preprocess(df)

            # 保存数据
            save_dir = Path('data/processed/cicids2017')
            save_dir.mkdir(parents=True, exist_ok=True)

            np.save(save_dir / 'X_train.npy', X_train)
            np.save(save_dir / 'y_train.npy', y_train)
            np.save(save_dir / 'X_test.npy', X_test)
            np.save(save_dir / 'y_test.npy', y_test)

            print(f"\n数据已保存至 {save_dir}")


if __name__ == '__main__':
    main()
