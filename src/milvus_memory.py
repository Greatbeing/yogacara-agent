import time, logging, numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MilvusMemory:
    def __init__(self, config: Dict[str, Any]):
        self.host = config.get("host", "localhost"); self.port = config.get("port", 19530)
        self.collection_name = config.get("collection", "yogacara_seeds"); self.dim = config.get("dim", 11); self.capacity = config.get("capacity", 50000)
        connections.connect("default", host=self.host, port=self.port)
        if not utility.has_collection(self.collection_name):
            fields = [FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), FieldSchema(name="emb", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                      FieldSchema(name="act", dtype=DataType.VARCHAR, max_length=10), FieldSchema(name="rew", dtype=DataType.FLOAT), FieldSchema(name="ts", dtype=DataType.DOUBLE),
                      FieldSchema(name="imp", dtype=DataType.FLOAT), FieldSchema(name="align", dtype=DataType.FLOAT), FieldSchema(name="unc", dtype=DataType.FLOAT), FieldSchema(name="tag", dtype=DataType.VARCHAR, max_length=20)]
            schema = CollectionSchema(fields, description="Yogacara Seed Memory")
            self.collection = Collection(self.collection_name, schema)
            self.collection.create_index("emb", {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 16, "efConstruction": 64}})
        else:
            self.collection = Collection(self.collection_name)
        self.collection.load()

    def add(self, seed: Dict[str, Any]):
        self.collection.insert([[seed["emb"]], [seed["act"]], [seed["rew"]], [seed["ts"]], [seed["imp"]], [seed["align"]], [seed["unc"]], [seed["tag"]]])
        if self.collection.num_entities > self.capacity: self.collection.delete("imp < 0.15"); self.collection.compact()

    def retrieve(self, obs_emb: List[float], k: int = 3, min_imp: float = 0.1) -> List[Dict]:
        res = self.collection.search(data=[obs_emb], anns_field="emb", param={"metric_type": "L2", "params": {"ef": 32}}, limit=k, expr=f"imp >= {min_imp}", output_fields=["act","rew","ts","imp","align","unc","tag"])
        return [{"act": h.entity.get("act"), "rew": h.entity.get("rew"), "ts": h.entity.get("ts"), "imp": h.entity.get("imp"), "align": h.entity.get("align"), "unc": h.entity.get("unc"), "tag": h.entity.get("tag")} for hits in res for h in hits]

    def perfume_update(self, decay_rate: float = 0.12, reward_boost: float = 0.3):
        now = time.time(); res = self.collection.query("id > 0", output_fields=["id","ts","rew","imp"])
        if not res: return
        ids, new_imps = [], []
        for row in res:
            dt = now - row["ts"]; imp = row["imp"] * np.exp(-decay_rate * dt); imp = min(1.0, imp + reward_boost * max(0, row["rew"]))
            ids.append(row["id"]); new_imps.append(imp)
        self.collection.upsert([ids, new_imps], ["id", "imp"]); logger.info(f"🔄 Milvus熏习完成: 更新 {len(ids)} 条")
