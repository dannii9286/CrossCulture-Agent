# CrossCulture-Agent 🌍

> 基于发音相似度、结构化知识检索与 Multi-stage RAG 的跨文化姓名推荐 Agent

## 📖 项目简介

CrossCulture-Agent 是一个面向跨语言、跨文化场景的智能姓名推荐系统。

传统大语言模型在进行“中文名 → 英文 / 法文 / 日文名”推荐时，容易出现以下问题：

- **发音断层**：推荐名字与原中文名读音差异较大
- **文化幻觉**：LLM 可能生成缺乏可靠依据的词源或文化解释
- **文化适配不足**：忽略目标国家的真实命名习惯
- **俚语与禁忌风险**：名字可能存在负面谐音、俚语或文化歧义
- **结果不可解释**：难以说明名字为什么被推荐

因此，本项目将姓名推荐拆解为多个可验证模块，通过：

**Phonetic Analysis → Candidate Retrieval → Hybrid RAG → Risk Filtering → LLM Explanation**

实现兼顾发音、语义、文化适配与安全性的跨文化姓名推荐。

---

## 🏗️ 系统架构

```text
                    User Request
                         │
                         ▼
                 FastAPI / Pydantic
                         │
                         ▼
                Phonetic Analysis
             中文名 → Pinyin → Normalize
                         │
                         ▼
               Candidate Retrieval
          Culture / Gender / Letter Filter
                         │
                         ▼
              Hybrid Retrieval（开发中）
                 ┌───────┴───────┐
                 ▼               ▼
              Dense             BM25
             Retrieval         Retrieval
                 └───────┬───────┘
                         ▼
                        RRF
                         │
                         ▼
              Phonetic Re-ranking
                         │
                         ▼
              Anti-Taboo Filter（规划）
                         │
                         ▼
                LLM Agent（规划）
                         │
                         ▼
              Structured JSON Response
```

---

## ✨ 当前已实现功能

### 1. FastAPI 后端服务

使用 FastAPI + Pydantic v2 构建 REST API：

```text
POST /api/v1/generate-names
```

支持输入：

- 中文姓名
- 性别
- 目标文化
- 个性 / 风格标签
- 首字母偏好
- 推荐数量等约束

并返回结构化 JSON 结果。

---

### 2. 中文姓名发音分析

使用 `pypinyin` 将中文姓名转换为标准拼音，并进行统一的 Phonetic Normalization。

例如：

```text
林雨桐
   ↓
lin yu tong
   ↓
linyutong
```

当前使用确定性算法而非 LLM 对发音相似度进行打分。

### Phonetic Similarity

使用 Levenshtein Distance：

```text
similarity =
1 - levenshtein_distance(a, b) / max(len(a), len(b))
```

返回范围：

```text
0.0 ~ 1.0
```

其中：

- `1.0`：拼写 / 发音表示高度接近
- `0.0`：差异较大

示例：

| 中文名 | 候选名 | Similarity |
|---|---|---:|
| 雨娜 | Yuna | 1.000 |
| 雨娜 | Clara | 0.200 |
| 李明 | Liming | 1.000 |
| 林雨桐 | Lin Yutong | 1.000 |
| 张伟 | Christopher | 0.182 |

> 当前版本使用拼音字符串作为可解释的 baseline，后续计划升级为 IPA / 多语言 G2P 音素级相似度。

---

### 3. Multi-Culture Name Knowledge Base

当前建立了第一版本地姓名知识库：

```text
data/names.csv
```

共包含 **66 条姓名数据**，覆盖：

| Culture | 数量 |
|---|---:|
| en-UK | 15 |
| en-US | 15 |
| fr | 20 |
| ja | 16 |

每个候选姓名包含：

```text
name
culture
gender
ipa
origin
etymology
meaning
style_tags
popularity
```

对于无法可靠确认的 IPA、词源或含义字段保持为空，避免人为制造错误文化信息。

---

### 4. Candidate Retrieval

当前实现基于结构化规则的候选检索 Pipeline：

```text
Chinese Name
     │
     ▼
Culture Filter
     │
     ▼
Gender Filter
     │
     ▼
Preferred Letter
     │
     ▼
Phonetic Similarity
     │
     ▼
Personality Match
     │
     ▼
Final Ranking
```

当前 Baseline Ranking：

```text
Final Score =
    0.7 × Phonetic Similarity
  + 0.2 × Personality Tag Match
  + 0.1 × Popularity
```

所有权重统一存放于配置模块中，避免在业务逻辑中 Hard Code。

对于首字母偏好采用软约束：

```text
Preferred Letter Match
        │
       Yes ──→ 优先返回
        │
        No
        ↓
候选数量不足时自动 fallback
```

避免由于用户约束过强导致无结果返回。

---

## 🚧 开发中：Hybrid RAG

下一阶段将候选检索升级为：

```text
                 Query
                   │
          ┌────────┴────────┐
          ▼                 ▼
    Dense Retrieval       BM25
    Semantic Search    Sparse Search
          │                 │
          └────────┬────────┘
                   ▼
          Reciprocal Rank Fusion
                   │
                   ▼
            Candidate Pool
                   │
                   ▼
          Phonetic Re-ranking
```

### Dense Retrieval

使用 multilingual Sentence Transformer Embedding，对：

- Personality
- Meaning
- Style
- Cultural Context

进行语义匹配。

### BM25

用于：

- Name
- Culture
- Gender
- Style Tags
- Keyword

等 lexical / exact matching。

### RRF

使用 Reciprocal Rank Fusion 融合 Dense 与 Sparse Retrieval：

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

避免直接融合 BM25 与 Cosine Similarity 不同尺度的原始分数。

---

## 🛡️ Roadmap：Anti-Taboo Risk Engine

计划增加跨文化姓名风险检测：

```text
Candidate Name
      │
      ▼
Exact Slang Match
      │
      ▼
Fuzzy Match
      │
      ▼
Phonetic Slang Match
      │
      ▼
Semantic Risk Check
      │
      ▼
LLM Fallback
```

采用 Cascade Routing 思路：

- **低风险** → 规则直接通过
- **高风险** → 直接拦截
- **不确定样本** → 路由至 LLM 深度判断

减少不必要的 LLM 调用，并提升风险判断的可解释性。

---

## 🤖 Roadmap：LLM Agent

LLM 不直接从零生成姓名，而主要负责：

- 综合 RAG Evidence
- 解释 Cultural Fit
- 生成结构化推荐理由
- 总结已检索到的词源信息
- 处理规则无法确定的复杂文化语境

核心原则：

> **Retrieve First, Generate Second**

候选姓名与事实信息优先来自知识库和检索结果，降低 LLM Cultural Hallucination 风险。

---

## 🛠️ 技术栈

### 当前使用

```text
Python 3.10+
FastAPI
Pydantic v2
Asyncio
pypinyin
Pandas
Pytest
```

### 开发中 / 规划

```text
Sentence-Transformers
BM25 / rank-bm25
FAISS / Milvus
Redis
OpenAI API / DeepSeek
Docker
```

---

## 📁 项目结构

```text
CrossCulture-Agent/
│
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── routes.py
│   │
│   ├── config/
│   │   └── constants.py
│   │
│   ├── models/
│   │   ├── schemas.py
│   │   └── candidate.py
│   │
│   ├── services/
│   │   ├── phonetic.py
│   │   ├── candidate_retriever.py
│   │   ├── rag_engine.py
│   │   ├── taboo_filter.py
│   │   └── llm_engine.py
│   │
│   └── main.py
│
├── data/
│   └── names.csv
│
├── tests/
│   ├── test_api.py
│   ├── test_phonetic.py
│   └── test_candidate_retriever.py
│
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

---

## 🚀 快速开始

### 1. Clone 项目

```bash
git clone <your-repository-url>
cd CrossCulture-Agent
```

### 2. 创建虚拟环境

Windows：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 启动服务

```bash
uvicorn app.main:app --reload
```

启动成功后访问：

```text
http://127.0.0.1:8000/docs
```

即可通过 Swagger UI 调试 API。

---

## 📡 API 示例

### Request

```json
{
  "chinese_name": "雨娜",
  "gender": "female",
  "target_cultures": ["ja"],
  "personality_tags": ["bright"],
  "constraints": {
    "preferred_letters": ["Y"],
    "max_recommendations": 3
  }
}
```

### Candidate Ranking

当前 Baseline 可以得到类似：

```text
Yuna      0.9920
Hina      0.6340
Hinata    0.5173
```

实际结果由当前知识库和 Ranking Pipeline 决定。

---

## 🧪 测试

运行：

```bash
python -m pytest -q
```

当前测试结果：

```text
20 passed
```

测试覆盖：

- API Request / Response
- Pydantic Schema
- Pinyin Normalization
- Levenshtein Similarity
- Culture Filtering
- Gender Filtering
- Preferred Letter
- Fallback Strategy
- Personality Tag Matching
- Candidate Ranking
- Top-K Constraint

---

## 📊 当前项目进度

```text
[✓] FastAPI Backend
[✓] Pydantic Schema
[✓] Phonetic Normalization
[✓] Levenshtein Similarity
[✓] Multi-Culture Name Knowledge Base
[✓] Candidate Retrieval
[✓] Baseline Ranking
[✓] Automated Tests

[ ] Dense Retrieval
[ ] BM25 Retrieval
[ ] Reciprocal Rank Fusion
[ ] Retrieval Evaluation
[ ] Anti-Taboo Risk Engine
[ ] Redis Cache
[ ] LLM Structured Generation
[ ] Docker Deployment
```

---

## 🎯 项目设计原则

CrossCulture-Agent 的核心目标并非简单调用 LLM 生成姓名，而是将跨文化命名拆解成多个**可检索、可验证、可解释、可测试**的工程模块：

```text
LLM-only Naming
        ↓
Retrieval + Deterministic Scoring
        ↓
Risk Control
        ↓
LLM-assisted Explanation
```

通过减少对 LLM 自由生成能力的依赖，提高跨文化姓名推荐的真实性、稳定性和可解释性。

---

## 📌 当前局限

当前版本仍处于 MVP 阶段：

- 姓名知识库规模较小
- 拼音字符串距离不等价于真实语音距离
- 暂未实现完整 IPA / 多语言 G2P
- Style Tags 存在人工标注主观性
- Popularity 暂不支持跨文化直接比较
- Hybrid RAG 与 Anti-Taboo 模块仍在开发中

后续将持续扩展知识库规模，并完善 Hybrid Retrieval、音素级匹配、风险控制和 LLM Agent Pipeline。
