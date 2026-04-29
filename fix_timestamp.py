"""Fix timestamp collision bug in yogacara_test.py"""

with open(r'src/yogacara_agent/yogacara_test.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Seed dataclass field name
content = content.replace(
    'timestamp: float = 0.0',
    'timestamp_ns: int = 0  # nanosecond-precision unique ID'
)

# Fix 2: Seed creation uses time.time_ns()
content = content.replace(
    'time.time(),\n                0.8,',
    'time.time_ns(),\n                0.8,'
)

# Fix 3: to_dict() uses timestamp_ns
content = content.replace(
    '"ts": self.timestamp,\n            "imp":',
    '"ts": self.timestamp_ns,\n            "imp":'
)

# Fix 4: ts_to_seed keyed by timestamp_ns
content = content.replace(
    'ts_to_seed = {s.timestamp: s for s in self.seeds}',
    'ts_to_seed = {s.timestamp_ns: s for s in self.seeds}'
)

# Fix 5: dict seeds use "ts_ns" key to avoid collision with timestamp field
# No, dict seeds already use "ts". The issue is that when we call seed.to_dict()
# we get the nanosecond ts. But when VipakaFeedback calls batch_update, the dicts
# have "ts" from to_dict(). This should work. But let's verify:
# Actually the real fix is to make all timestamp references consistent.
# Let's also update the consolidation.remove_seeds_by_ts to work with nanoseconds.

# Fix 6: remove_seeds_by_ts - it's already fine, it just needs nanosecond ts values
# which to_dict() now provides.

# Fix 7: Also update the vipaka batch_update call to pass correct ts
# Let me check what vipaka.process_step does:
# It calls self.alaya.batch_update(dict_seeds) where dict_seeds come from to_dict()
# which now returns timestamp_ns as "ts". This should work.

with open(r'src/yogacara_agent/yogacara_test.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fix applied")
