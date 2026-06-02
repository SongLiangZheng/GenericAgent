## PIC MVP

- Starlette 1.0 下 `TemplateResponse` 调用顺序需用 `(request, name, context)`；旧写法会把模板名/上下文错位，导致 UI 页 500。
- `POST /api/v1/auth/login` 使用 `OAuth2PasswordRequestForm`，必须提交 `application/x-www-form-urlencoded` 的 `username/password`；发 JSON 会 422。
- 人才库 API 路由已验证为 `/api/v1/talent`，不是 `/api/v1/talents`。
- UI 未登录访问 `/ui/talent`、`/ui/sync`` 时可返回 200 但实际内容是登录页；判断权限不能只看状态码。
- `AIParser.parse_resume()` / `generate_portrait()` 是 async 方法，调用必须加 `await`，否则返回 coroutine 对象静默失败。
- `sync_service._get_or_create_candidate` 默认不拉详情；需传 `fetch_detail=True` 才调用 `get_candidate_detail` 写入 `resume_text`；`sync_applications` 需手动更新 `Candidate.current_stage/current_job_title/department`。
- `ui_router.py` 模板渲染前须将 ORM 对象转为 dict，`datetime` 字段格式化为字符串，否则 Jinja2 渲染报错。
- **Alembic 初始化**：`alembic.ini` 禁用中文注释（Windows GBK 解码失败）；`env.py` 须用 async 模式（`run_async_migrations`）；`env.py` 须 import 所有 models 才能让 `Base.metadata` 有表。
- **Alembic autogenerate 对已有 DB 返回 pass**：因表已存在，需用临时 DB 路径（`DATABASE_URL=sqlite+aiosqlite:///./tmp_alembic.db alembic revision --autogenerate`），生成后 stamp 现有 DB（`alembic stamp head`）。
- **Windows pip 路径不一致**：`pip install` 可能装到不同 Python；始终用 `python -m pip install`。
- **DB 被服务器锁定无法移动**：不要 mv/rename pic.db，改用临时 DATABASE_URL 环境变量绕过。
- **PIC SQLite 真实业务表名**：已验证为 `candidates` / `applications` / `interviews`；不要误查 `talents`，否则会误判“无数据/无此表”。
- **浏览器验 PIC 列表页别首屏定论**：`/ui/talent` 可能停留旧 DOM 显示 0 条；强制刷新或带时间戳重开后再判断，随后再进 `/ui/talent/{id}` 核验详情中的申请/面试/AI画像。
- **本地 8013 启动失败先查依赖**：已验证缺少 `aiosqlite` 时服务会直接起不来；安装后可恢复监听。
- **PDF 预览需区分“无模块”与“无附件”**：已验证详情页会出现“简历预览”区块；若当前仅同步到简历文本、没有 PDF/附件 URL，则显示“当前仅同步到简历文本，尚无可预览的 PDF/附件链接”，且不会出现 iframe。
- **PIC 本地演示数据可直接写真实业务表闭环验证**：已验证 `data/talent.db` 的 `candidates/applications/interviews` 可直接写入测试候选人、申请和面试数据；`/api/v1/talent/{id}` 会返回 `applications/interviews`，详情页 PDF 预览由 `candidates.resume_url/resume_filename/resume_file_id` 驱动。 