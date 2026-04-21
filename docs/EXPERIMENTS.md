# 🧪 实验手册
## 1. 自动化实验
```bash
python src/exp_automator.py
# 输出: experiments/*.csv + fig1_reward_ci.pdf
```
## 2. 奖励塑形
修改 `src/reward_designer.py` 配置势函数与课程学习。
> ✅ `F = γΦ(s') - Φ(s)` 保证不改变 MDP 最优策略。

## 3. 消融设计
| 组别 | 移除模块 | 预期变化 |
|------|----------|----------|
| Full | 无 | 基线 |
| w/o Manas | 第七识 | 拦截0% / 陷阱命中↑ |
| w/o Alaya | 第八识 | 无收敛 |
| w/o SlowLoop | 慢循环 | 经验遗忘 |

## 4. 可重复性
固定种子 `random.seed(42)`，记录依赖哈希与 Git SHA。
