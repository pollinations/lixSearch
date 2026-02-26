┌─────────────────────────────────────────────────────────────┐
│         Shared Infrastructure Layer                         │
│                                                             │
│  Embedding Model (Mounted Volume):                          │
│  /shared/embedding_models/all-MiniLM-L6-v2                 │
│  ├─ Size: 500MB (ONE COPY, all containers read it)        │
│  └─ Read-only mount for all 10 containers                 │
│                                                             │
│  Embeddings Storage (Shared Disk/NFS):                      │
│  /shared/data/embeddings/                                  │
│  ├─ Chroma DB vectors (all containers write/read)          │
│  ├─ Size: 50GB (computed once, used by all)               │
│  └─ NFS or EBS mount                                       │
│                                                             │
│  Redis (Single Instance):                                  │
│  redis://redis-service:6379                               │
│  ├─ Single source of truth for all containers             │
│  └─ All 10 containers connect to it                       │
└────────────┬──────────┬──────────┬────────────┬────────────┘
            │          │          │            │
       ┌────▼──┐  ┌───▼──┐  ┌───▼──┐    ┌───▼──┐
       │ Cont1 │  │ Cont2│  │ Cont3│ .. │ Cont10
       │ Reads │  │ Reads│  │ Reads│    │ Reads
       │ model │  │ model│  │ model│    │ model
       │ share │  │ share│  │ share│    │ share
       └───────┘  └──────┘  └──────┘    └──────┘

Memory usage:
- Embedding model: ~500MB (ONE COPY, shared via mount!) ✅
- Per-container overhead: 1-2GB × 10 = 10-20GB
- Redis: ~2GB (one instance)
- Embeddings: ~50GB (one copy on shared storage)
─────────────────────────────
TOTAL: ~63-73GB (vs 525GB with separate!)
SAVED: 450GB+