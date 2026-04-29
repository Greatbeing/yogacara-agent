"""Fix timestamp bug: prune_ts uses timestamp_ns not timestamp"""
with open(r'src/yogacara_agent/yogacara_test.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix prune_ts: use timestamp_ns instead of timestamp
content = content.replace(
    'prune_ts = [s.timestamp for s in all_prune]',
    'prune_ts = [s.timestamp_ns for s in all_prune]'
)

# Fix remove_seeds_by_ts: compare with timestamp_ns
content = content.replace(
    'self.seeds = [s for s in self.seeds if s.timestamp not in ts_set]',
    'self.seeds = [s for s in self.seeds if s.timestamp_ns not in ts_set]'
)

with open(r'src/yogacara_agent/yogacara_test.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done - timestamp_ns fix applied")
