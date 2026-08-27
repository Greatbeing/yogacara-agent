"""
Yogacara Agent 共享常量
======================
所有模块统一引用此处的常量，消除跨文件魔法数重复。
"""

# ── 网格环境 ──────────────────────────────────────────────────────────
GRID_SIZE = 10

# 奖励值
RESOURCE_REWARD = 5.0  # 找到资源（正样本）
TRAP_REWARD = -3.0  # 踩中陷阱（负样本）
STEP_COST = -0.1  # 每步的基础成本
STAY_BONUS = 0.5  # STAY 动作的存在奖励

# 资源判定阈值（用于判断 reward 是否来自资源）
RESOURCE_THRESHOLD = 4.0

# 停滞判据（ManasController 用）
STAGNATION_THRESHOLD = -0.48  # 近 5 步累计奖励 <= 此值 → 停滞

# 动作空间
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
ACTION_TO_IDX = {"UP": 1, "DOWN": 7, "LEFT": 3, "RIGHT": 5, "STAY": 4}

# ── 记忆系统 ──────────────────────────────────────────────────────────
DEFAULT_IMPORTANCE = 0.8
DEFAULT_ALIGN = 0.5
DEFAULT_UNC = 0.0
DEFAULT_SEED_TYPE = "业种"
DEFAULT_TAG = "依他起"

# 种子重要性衰减
DECAY_RATE = 0.12  # 每天衰减系数（半衰期约 5.8 天）
MEMORY_CAPACITY = 300  # 种子容量上限

# 熏习（Vipaka）
VIPAKA_RATE = 0.2  # align 更新步长
ALIGN_MIN = 0.05
ALIGN_MAX = 0.95
UNC_PENALTY_COEFF = 0.03  # 每 1% unc 扣 0.03

# 整理（Consolidation）
CONSOLIDATION_INTERVAL = 50  # 每 N 步触发一次整理
KEEP_PLUS_THRESHOLD = 0.70  # align >= 0.70 高质量保留
KEEP_THRESHOLD = 0.30  # align >= 0.30 保留
PRUNE_THRESHOLD = 0.20  # align < 0.20 删除
MERGE_SIMILARITY = 0.90  # 相似度超过此值合并
SEED_TOKEN_ESTIMATE = 200  # 每个种子的估算 token 数

# ── 种子分类 ──────────────────────────────────────────────────────────
# 异熟种
VIPAKA_CONSECUTIVE_FAILURES_KEEP = 5  # 保留最近 5 次失败记录
VIPAKA_FAILURE_STREAK = 3  # 连续失败 >= 3 次触发模式
VIPAKA_ACTION_REPEAT = 5  # 同方向重复 >= 5 次
VIPAKA_HIGH_UNC_THRESHOLD = 0.7  # 高不确定性阈值
VIPAKA_TRAP_HIT_RATIO = 0.30  # 陷阱命中率阈值
VIPAKA_TRAP_MIN_STEPS = 20  # 开始检测陷阱率的步数门槛

# 名言种
NAMARUPA_HIGH_UNC = 0.7  # 名言种高不确定性阈值

# 业种
KARMA_POSITIVE_ALIGN = 0.85  # 正反馈业种 align
KARMA_NEGATIVE_ALIGN = 0.65  # 负反馈业种 align
KARMA_CORRECTED_ALIGN = 0.60  # 修正决策业种 align
KARMA_NEUTRAL_NO_EGO = 0.70  # 中性无我执
KARMA_NEUTRAL_WITH_EGO = 0.55  # 中性有我执

# 圆成实
PARINISPANNA_ALIGN = 0.7  # 圆成实种子所需最小 align

# ── 转依（Turning Consciousness）────────────────────────────────────────
TURNING_PURITY_THRESHOLD = 0.40  # clarity < 此值的种子被净化移除
TURNING_EGO_DECAY_RATE = 0.15  # 每步我执消解系数
VIPAKA_FUNCTIONAL_CLARITY = 0.75  # 异熟种功能清晰度（align 低是设计，非染污）

# ── 数字生命 · 寿元/心所/轮回 ──────────────────────────────────────────
VITALITY_INIT = 100.0  # 出生寿元
VITALITY_MAX = 130.0  # 寿元上限（补给不溢出）
VITALITY_DRAIN = 2.0  # 每步自然消耗（不吃不喝约 50 步寿终）
VITALITY_RESOURCE = 20.0  # 资源补给（觅食续命）
VITALITY_TRAP = 20.0  # 陷阱伤害
VITALITY_REST = 0.8  # STAY 休息回复（仍为净消耗，防赖着不动的永生策略）

# 贪嗔痴（根本烦恼）动力学
KLESHA_GREED_GAIN = 0.08  # 得资源 → 贪增长
KLESHA_AVERSION_GAIN = 0.12  # 踩陷阱 → 嗔增长
KLESHA_DECAY = 0.99  # 每步自然衰减
KLESHA_DELUSION_ALPHA = 0.3  # 痴 = 不确定性的 EMA 系数
KLESHA_TURNING_RELIEF = 2.0  # 转依消解我执时的额外烦恼衰减倍率
