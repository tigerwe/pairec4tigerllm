# -*- coding: utf-8 -*-
"""
Python 推理客户端.

用于调用生成式召回推理服务.
"""

import json
import time
from typing import List, Dict, Optional
import requests


class GenerativeRecallClient:
    """生成式召回客户端.

    封装对推理服务的 HTTP 调用.
    """

    def __init__(
        self,
        server_url: str = 'http://localhost:8000',
        timeout: float = 5.0,
        max_retries: int = 3
    ):
        """初始化客户端.

        Args:
            server_url: 推理服务地址
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries

    def health_check(self) -> bool:
        """健康检查.

        Returns:
            服务是否健康
        """
        try:
            response = requests.get(
                f'{self.server_url}/health',
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception:
            return False

    def recommend(
        self,
        user_id: str,
        history: List[List[int]],
        topk: int = 10,
        temperature: float = 1.0,
        beam_width: int = 1
    ) -> Dict:
        """获取推荐.

        Args:
            user_id: 用户 ID
            history: 用户历史语义 ID 序列
            topk: 推荐数量
            temperature: 采样温度
            beam_width: Beam search 宽度

        Returns:
            推荐结果

        Raises:
            ConnectionError: 连接失败
            RuntimeError: 服务返回错误
        """
        payload = {
            'user_id': user_id,
            'history': history,
            'topk': topk,
            'temperature': temperature,
            'beam_width': beam_width
        }

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f'{self.server_url}/recommend',
                    json=payload,
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'}
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    error_msg = response.json().get('error', 'Unknown error')
                    raise RuntimeError(f"Service error: {error_msg}")

            except requests.exceptions.Timeout:
                last_exception = ConnectionError(f"Request timeout (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError as e:
                last_exception = ConnectionError(f"Connection failed: {e}")
            except Exception as e:
                last_exception = e

            if attempt < self.max_retries - 1:
                time.sleep(0.5 * (attempt + 1))  # 指数退避

        raise last_exception

    def batch_recommend(
        self,
        requests: List[Dict]
    ) -> List[Dict]:
        """批量获取推荐.

        Args:
            requests: 请求列表，每个元素是字典包含 user_id, history 等

        Returns:
            推荐结果列表
        """
        results = []
        for req in requests:
            try:
                result = self.recommend(**req)
                results.append(result)
            except Exception as e:
                results.append({
                    'code': 500,
                    'error': str(e)
                })
        return results


def main():
    """测试客户端."""
    import argparse

    parser = argparse.ArgumentParser(description='Test generative recall client')
    parser.add_argument('--server_url', type=str, default='http://localhost:8000',
                        help='Server URL')
    parser.add_argument('--user_id', type=str, default='test_user',
                        help='User ID')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of recommendations')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')

    args = parser.parse_args()

    # 创建客户端
    client = GenerativeRecallClient(server_url=args.server_url)

    # 健康检查
    print("Checking server health...")
    if not client.health_check():
        print("Server is not healthy!")
        return
    print("Server is healthy")

    # 测试推荐
    print(f"\nGetting recommendations for user {args.user_id}...")

    # 模拟历史记录（4 层语义 ID）
    history = [
        [100, 50, 25, 10],
        [101, 51, 26, 11],
        [102, 52, 27, 12]
    ]

    try:
        result = client.recommend(
            user_id=args.user_id,
            history=history,
            topk=args.topk,
            temperature=args.temperature
        )

        print(f"\nRecommendations:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
