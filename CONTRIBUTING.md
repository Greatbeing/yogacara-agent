# 🤝 贡献指南
感谢关注 **Yogacara Agent**！本项目遵循 Apache 2.0 协议。

## 📋 提交前检查
- [ ] 代码已通过 `pre-commit run --all-files`
- [ ] 新增功能附带单元测试
- [ ] 文档已同步更新
- [ ] 提交信息遵循 Conventional Commits

## 🛠️ 本地开发
```bash
git clone https://github.com/Greatbeing/yogacara-agent.git
cd yogacara-agent
pip install -e ".[dev]"
pre-commit install
```

## 🔄 PR 流程
1. Fork 并创建分支 `git checkout -b feat/xxx`
2. 提交并推送 `git push origin feat/xxx`
3. 提交 PR，填写模板并关联 Issue

提交 PR 即表示同意以 Apache 2.0 协议开源。
