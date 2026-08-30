# 筑审智核（BuildCheck-Agent）

筑审智核是面向建筑施工方案的智能处理工具。V0.2 在稳定的 V0.1 PDF 上传与文本解析基线之上，新增施工方案智能理解和结构化信息提取；本版本不进行方案审查、评分或整改建议生成。

## 当前版本：V0.2.1；V0.3 Development Candidate

V0.3 正在开发施工方案完整性审查能力。在复用 V0.2.1 结构化分析结果的基础上，检查方案关键组成部分是否存在、证据位于何处，以及内容缺失或覆盖不足时建议补充什么。V0.3 Development Candidate 不是规范合规审查，不提供法律结论或施工风险评级。

开发中的公开接口：

```text
POST /api/v1/documents/{document_id}/review/completeness?force=false
```

`force` 只控制完整性 review cache，不会强制重新执行文档 LLM analysis。

V0.1 基线能力保持兼容：

- PDF 文件上传（最大 30 MB）
- PyMuPDF 完整文本及逐页文本提取
- 文件类型、空文件、损坏 PDF、无文本层 PDF 校验
- `GET /health`
- `POST /api/v1/documents/upload`

V0.2 新增能力：

- PDF 文本解析及保留页码的保守文本预处理
- 文档类型识别、工程类别/专业识别和项目元数据提取
- 专项施工工期提取及多级章节结构识别
- 编制依据及标准规范名称与编号提取（只记录引用，不判断有效性或符合性）
- `main_work_items`、`main_materials` 和 `main_methods` 提取
- 质量、安全、环境和应急章节存在性识别
- OpenAI Chat Completions 风格兼容 LLM 接口
- 超长文档按配置的字符上限分段提取并合并
- JSON 输出清洗、Pydantic 验证及 `source_page`、`source_text` 原文溯源
- 分析结果 JSON 文件缓存和 `force` 强制刷新

真实样本验证已覆盖给排水专项施工方案、智能化弱电施工组织设计、道路高边坡专项施工方案，以及深基坑支护及开挖专项施工方案。V0.2 具备初步跨模板、跨专业泛化能力，但不声称支持所有施工专业。

V0.2 明确不包含规范审查、风险评级、方案评分、整改建议、RAG、向量数据库、OCR、数据库、前端或报告生成。

> 扫描版 PDF 仍不支持 OCR，仅处理带有可提取文本层的 PDF。

## 环境要求

- Python 3.11
- Windows PowerShell（以下示例命令）

## 安装与运行

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Set-Location backend
uvicorn app.main:app --reload
```

服务启动后可访问：

- 健康检查：<http://127.0.0.1:8000/health>
- Swagger：<http://127.0.0.1:8000/docs>
- OpenAPI：<http://127.0.0.1:8000/openapi.json>

## LLM 配置

复制 `.env.example` 中的变量到运行环境使用的 `.env`，或直接设置环境变量：

```dotenv
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://your_llm_provider.example/v1
LLM_MODEL=your_model_name_here
LLM_TIMEOUT=60
LLM_MAX_INPUT_CHARS=60000
```

应用会从项目根目录加载 `.env`，已有环境变量优先；也可由进程管理器、容器或 shell 注入这些变量。`.env` 已被 Git 忽略，真实密钥不得写入代码、文档或测试。`LLM_BASE_URL` 应指向兼容 API 根路径（例如以 `/v1` 结尾），服务会请求其 `/chat/completions`。

## API

### 健康检查

```text
GET /health
```

### 上传并解析 PDF

```text
POST /api/v1/documents/upload
Content-Type: multipart/form-data
表单字段：file
```

服务器用 UUID 保存文件，上传响应中的 `document_id` 用于后续分析。

### 分析已上传文档

```text
POST /api/v1/documents/{document_id}/analyze?force=false
```

- `force=false`（默认）：存在 `backend/data/analysis/{document_id}.json` 时直接返回，`cached=true`。
- `force=true`：重新提取和分析，并原子覆盖缓存。
- 文档不存在返回 404；无有效文本返回 422；LLM 未配置返回 503；LLM 调用或输出失败返回 502。

响应中的关键字段尽量包含 `value`、`source_page` 和简短 `source_text`。来源无法确认时页码和原文为 `null`，不得推测来源。

## 运行测试

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests -v
```

自动化测试动态生成 PDF，并通过依赖注入使用 Fake LLM，不依赖互联网、真实模型或 API Key。

## 文件存储

- 上传 PDF：`backend/data/uploads/{document_id}.pdf`
- 分析缓存：`backend/data/analysis/{document_id}.json`

两个目录中的运行时文件均被 Git 忽略，仅提交 `.gitkeep`。

## V0.2 已知限制

- 不支持扫描件 OCR、加密 PDF、版面/图片语义分析。
- 章节识别采用基础标题规则，复杂目录或异常排版可能需要 LLM 语义信息辅助。
- 分段提取是简单可靠的顺序分段和字段合并，不是 Agent 或复杂 MapReduce。
- LLM 输出只用于提取原文明确事实，结果质量仍受原 PDF 文本层和模型能力影响。
