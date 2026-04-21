# 🚀 部署指南
## 1. 本地验证
```bash
python src/yogacara_test.py
python src/yogacara_langgraph.py
```
## 2. Docker
```bash
docker compose up -d --build
curl http://localhost:8000/health
```
## 3. K8s + Helm
```bash
helm install yogacara ./helm/yogacara-agent
kubectl get hpa yogacara-hpa -w
```
## 4. Ray + vLLM
```bash
serve run src.vllm_ray_topology:router_dep
```
## 故障排查
| 现象 | 命令 | 解决 |
|------|------|------|
| CrashLoop | `kubectl logs -f deploy/yogacara-agent` | 检查 ConfigMap/密钥 |
| vLLM OOM | `nvidia-smi` | 调低 `gpu_memory_utilization` |
| 指标缺失 | `curl localhost:8000/metrics` | 确认路由注册 |
