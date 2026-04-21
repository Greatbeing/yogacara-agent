# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-21

### Added
- Initial release of Yogacara Agent framework
- 八识认知架构工程化实现 (Eight-consciousness cognitive architecture)
  - 前五识: 多模态感知编码器
  - 第六识: LLM规划器 + 工作记忆
  - 第七识: 末那控制器 (元认知/安全过滤)
  - 第八识: 阿赖耶记忆 (种子库/在线熏习)
- 快慢双循环决策系统
- 三性判别机制 (遍计所执/依他起/圆成实)
- MVP验证模块 (`src/yogacara_test.py`)
- 生产级 LangGraph 编排 (`src/yogacara_langgraph.py`)
- LLM接入模块支持 Qwen/Llama (`src/llm_planner.py`)
- Milvus 向量记忆持久化 (`src/milvus_memory.py`)
- DPO + LoRA 在线对齐 (`src/online_alignment.py`)
- 安全模块: 输入净化、沙箱、限流、记忆守卫 (`src/security/`)
- 环境适配器: ROS2、Unity、Isaac Sim (`src/env_adapters/`)
- 实验自动化与论文图表生成 (`src/exp_automator.py`)
- Kubernetes + Helm 云原生部署清单
- Ray Serve + vLLM 分布式推理拓扑
- CI/CD 流水线 (ruff, mypy, pytest, codecov)

### Documentation
- README.md 项目说明
- 部署指南 (docs/DEPLOYMENT.md)
- 安全规范 (docs/SECURITY.md)
- 实验手册 (docs/EXPERIMENTS.md)
- 贡献指南 (CONTRIBUTING.md)
- Citation 文件 (CITATION.cff)

### Tests
- 核心模块单元测试 (tests/test_core.py)

---

## Future Roadmap

### [1.1.0] - Planned
- Web UI dashboard for real-time monitoring
- Pre-trained model weights release
- Extended environment adapters
- Multi-agent collaboration mode

### [1.2.0] - Planned
- ROS2 real-robot integration
- Visualization tools for seed evolution
- Academic benchmark suite
