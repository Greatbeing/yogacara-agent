"""
阿赖耶识持久化实现
==================
支持文件系统(JSONL)和可选的向量存储(Chroma)两种后端。

使用方式:from alaya_persistent import PersistentAlayaMemory

    # 文件存储（默认）
    alaya = PersistentAlayaMemory(storage="file", path="memory/seeds.jsonl")

    # 向量存储（需安装 chromadb）
    alaya = PersistentAlayaMemory(storage="vector", path="memory/chroma_db")
"""

import json  # noqa: F401
import math  # noqa: F401
import os
import time
from typing import Any

# 可选的向量存储
try:
    import chromadb

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

GRID_SIZE = 10


class PersistentAlayaMemory:
    """
    特性:
    - 文件持久化: JSONL 格式，每行一个种子
    - 向量检索: 基于 Chroma 的语义相似度搜索（可选）
    - 自动保存: 每次 add() 后自动刷盘
    - 种子分类: 支持名言种/业种/异熟种标签过滤
    """

    def __init__(self, storage: str = "file", path: str = "memory/seeds.jsonl"):
        """
        Args:
            storage: "file" | "vector" | "hybrid"
            path: 存储路径（文件路径或目录路径）
        """
        self.storage = storage
        self.path = path
        self.seeds: list[dict] = []
        self._chroma_client: Any = None
        self._chroma_collection: Any = None
        # 初始化存储
        if storage in ("file", "hybrid"):
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            self._load_from_file()

        if storage in ("vector", "hybrid"):
            if not HAS_CHROMA:
                raise ImportError("Chroma not installed. Run: pip install chromadb")
            self._init_chroma()

    # ── 编码与距离 ──────────────────────────────────────────────────────
    def _encode(self, obs: dict) -> list[float]:
        """将观察编码为向量。"""
        return [obs["pos"][0] / GRID_SIZE, obs["pos"][1] / GRID_SIZE] + [v / 2.0 for v in obs["grid_view"]]

    def _dist(self, a: list[float], b: list[float]) -> float:
        """欧氏距离。"""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    # ── 核心接口 ────────────────────────────────────────────────────────
    def retrieve(self, obs: dict, k: int = 3, seed_type: str | None = None) -> list[dict]:
        """
        检索相关种子。

        Args:
            obs: 当前观察
            k: 返回数量
            seed_type: 过滤特定种子类型（名言种/业种/异熟种）
        """
        if not self.seeds:
            return []

        # 类型过滤
        candidates = self.seeds
        if seed_type:
            candidates = [s for s in self.seeds if s.get("seed_type") == seed_type]

        if not candidates:
            return []

        # 向量检索（如果可用）
        if self.storage in ("vector", "hybrid") and self._chroma_collection:
            return self._retrieve_chroma(obs, k, seed_type)

        # 回退到欧氏距离
        emb = self._encode(obs)
        scored = sorted([(self._dist(emb, s["emb"]), s) for s in candidates], key=lambda x: x[0])
        return [s for _, s in scored[:k]]

    def add(self, seed: dict) -> None:
        """添加种子并持久化。"""
        # 确保必要字段
        seed.setdefault("ts", time.time())
        seed.setdefault("imp", 0.8)
        seed.setdefault("align", 0.5)
        seed.setdefault("unc", 0.0)
        seed.setdefault("tag", "依他起")
        seed.setdefault("seed_type", "业种")  # 默认业种

        self.seeds.append(seed)

        # 文件持久化
        if self.storage in ("file", "hybrid"):
            self._append_to_file(seed)

        # 向量存储
        if self.storage in ("vector", "hybrid") and self._chroma_collection:
            self._add_to_chroma(seed)

    def batch_update(self, seeds: list[dict]) -> None:
        """
        批量更新多个种子（用于 VipakaEngine 的 align 更新）。

        内存中已修改，直接触发文件重写持久化。
        """
        if not seeds:
            return
        if self.storage in ("file", "hybrid"):
            self._save_all_to_file()

    def retrieve_by_tags(self, tags: list[str], k: int = 5) -> list[dict]:
        """
        按标签检索种子（模糊匹配 tag 字段）。

        Args:
            tags: 标签列表，如 ["名言_遍计", "业_正反馈"]
            k: 返回上限
        """
        if not tags or not self.seeds:
            return []
        matched = [
            s for s in self.seeds
            if any(t in s.get("tag", "") for t in tags)
        ]
        # 按 importance 降序
        matched.sort(key=lambda s: s.get("imp", 0), reverse=True)
        return matched[:k]

    def perfume_update(self) -> None:
        """熏习更新：衰减旧种子重要性，提升高奖励种子。"""
        now = time.time()
        modified = False

        for s in self.seeds:
            dt = now - s.get("ts", now)
            if dt <= 0 or dt > 86400 * 365:
                continue
            s["imp"] *= math.exp(-0.12 * dt)
            s["imp"] = min(1.0, s["imp"] + 0.3 * max(0, s.get("rew", 0)))
            modified = True

        # 如果修改了，重写文件
        if modified and self.storage in ("file", "hybrid"):
            self._save_all_to_file()

    def get_stats(self) -> dict[str, Any]:
        """获取记忆统计。"""
        type_counts: dict[str, int] = {}
        for s in self.seeds:
            t = s.get("seed_type", "未知")
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_seeds": len(self.seeds),
            "storage_type": self.storage,
            "path": self.path,
            "seed_types": type_counts,
            "avg_importance": sum(s.get("imp", 0) for s in self.seeds) / len(self.seeds) if self.seeds else 0,
        }

    # ── 文件存储实现 ────────────────────────────────────────────────────
    def _load_from_file(self) -> None:
        """从 JSONL 文件加载种子。"""
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        seed = json.loads(line)
                        # 确保 emb 是列表
                        if "emb" in seed and isinstance(seed["emb"], str):
                            seed["emb"] = json.loads(seed["emb"])
                        self.seeds.append(seed)
                    except json.JSONDecodeError:
                        continue
            print(f"[Alaya] 从 {self.path} 加载了 {len(self.seeds)} 个种子")
        except Exception as e:
            print(f"[Alaya] 加载失败: {e}")

    def _append_to_file(self, seed: dict) -> None:
        """追加单个种子到文件。"""
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                # 将 emb 转为可序列化格式
                seed_copy = dict(seed)
                if isinstance(seed_copy.get("emb"), list):
                    seed_copy["emb"] = json.dumps(seed_copy["emb"])
                f.write(json.dumps(seed_copy, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Alaya] 保存失败: {e}")

    def _save_all_to_file(self) -> None:
        """全量保存（用于 perfume_update 后）。"""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                for seed in self.seeds:
                    seed_copy = dict(seed)
                    if isinstance(seed_copy.get("emb"), list):
                        seed_copy["emb"] = json.dumps(seed_copy["emb"])
                    f.write(json.dumps(seed_copy, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Alaya] 全量保存失败: {e}")

    # ── 向量存储实现 ────────────────────────────────────────────────────
    def _init_chroma(self) -> None:
        """初始化 Chroma 向量数据库。"""
        self._chroma_client = chromadb.PersistentClient(path=self.path)  # type: ignore[union-attr]
        self._chroma_collection = self._chroma_client.get_or_create_collection(  # type: ignore[union-attr]
            name="alaya_seeds", metadata={"hnsw:space": "l2"}
        )  # type: ignore[union-attr]
        print(f"[Alaya] Chroma 向量存储已初始化: {self.path}")

    def _add_to_chroma(self, seed: dict) -> None:
        """添加种子到 Chroma。"""
        if not self._chroma_collection:
            return

        seed_id = f"seed_{seed.get('ts', time.time())}_{id(seed)}"
        self._chroma_collection.add(
            ids=[seed_id],
            embeddings=[seed["emb"]],
            metadatas=[
                {
                    "action": seed.get("act", ""),
                    "reward": seed.get("rew", 0),
                    "importance": seed.get("imp", 0.8),
                    "seed_type": seed.get("seed_type", "业种"),
                    "tag": seed.get("tag", "依他起"),
                }
            ],
        )

    def _retrieve_chroma(self, obs: dict, k: int, seed_type: str | None) -> list[dict]:
        """从 Chroma 检索。"""
        if not self._chroma_collection:
            return []

        emb = self._encode(obs)
        where_filter = {"seed_type": seed_type} if seed_type else None

        results = self._chroma_collection.query(
            query_embeddings=[emb],
            n_results=k,
            where=where_filter,
        )

        # 转换回种子格式
        seeds = []
        for i, meta in enumerate(results["metadatas"][0]):
            seeds.append(
                {
                    "emb": results["embeddings"][0][i] if "embeddings" in results else emb,
                    "act": meta.get("action", ""),
                    "rew": meta.get("reward", 0),
                    "imp": meta.get("importance", 0.8),
                    "seed_type": meta.get("seed_type", "业种"),
                    "tag": meta.get("tag", "依他起"),
                }
            )
        return seeds


# ── 兼容层 ────────────────────────────────────────────────────────────
class AlayaMemory(PersistentAlayaMemory):
    """
    兼容旧版 AlayaMemory 的别名。
    默认使用文件存储，路径为 memory/seeds.jsonl。
    """

    def __init__(self):
        super().__init__(storage="file", path="memory/seeds.jsonl")
