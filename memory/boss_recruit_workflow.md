# Boss招聘工作流 SOP

## 概述
通过BOSS直聘自动化筛选并主动联系合适候选人

## 工作流步骤

### 1. 打开招聘平台
- 确保浏览器已打开 boss.zhipin.com
- 切换到招聘管理tab

### 2. 候选人筛选入口
- 进入 **"推荐牛人"** 页面（系统推荐候选人入口）
- 左侧菜单路径：招聘管理 → 推荐牛人 → 推荐牛人列表
- URL: `https://www.zhipin.com/web/geek/recommend`

### 3. 筛选条件设置
- **当前职位**：点击「当前职位」下拉，选择要招聘的具体职位（系统按职位JD推荐匹配候选人）
- **处理状态标签**：
  - `未处理`：新推荐未操作的候选人（优先看这里）
  - `已感兴趣`：已标记感兴趣，可跟进打招呼
  - `不合适`：已排除，一般忽略
- 列表字段：候选人、当前职位、期望职位、期望薪资、活跃情况、推荐时间

### 4. 扫描候选人列表
- 使用 web_scan 获取候选人列表
- 提取候选人姓名、当前职位、期望薪资、活跃情况等信息

### 5. 评估候选人
- 逐一阅读简历摘要
- 评分标准：技能匹配度、工作经验、公司背景

### 6. 主动打招呼
- 对评分≥7分的候选人发送招聘消息
- 消息模板：见下方

### 7. 消息模板
```
您好！我在BOSS直聘上看到您的简历，觉得您的背景与我们正在招聘的[职位]非常匹配。
我们公司[简短介绍]，薪资范围[XXX]，如果您近期有考虑新机会，欢迎聊聊！
```

## 注意事项
- 每日打招呼有上限，注意控制频率
- 优先联系活跃度高（近期在线）的候选人
- 记录已联系候选人，避免重复

## 技术实现
- 使用 tmwebdriver_sop 中的浏览器自动化方法
- 使用 web_scan + web_execute_js 扫描页面元素
- 点击前检查元素是否可见/可点击

## 关键技术细节（已验证）
- **推荐牛人iframe**：`iframe[name="recommendFrame"]`，同源，用 `iframe.contentDocument` 直接访问
- **职位切换（推荐牛人页）**：⚠️ 职位下拉在iframe内；点击 `input.chat-job-search` 触发下拉 → 等500ms → iDoc内 `.ui-dropmenu.job-selecter-wrap .job-item` LI点击
- **候选人卡片**：`.candidate-card-wrap`（iDoc内查询）；`card.innerText` 包含姓名/年龄/学历/公司名/技能
- **打招呼按钮**：`button.btn.btn-greet`；打招呼后变为 `.btn-continue`（继续沟通）= 成功标志
- **打招呼弹窗关闭**：成功后顶层document弹出 `.similar-geek-wrap`，点 `.iboss-close` 关闭
- **简历API推荐牛人页**：⚠️ `/wapi/zpgeek/resume/coach/resume.json` 在推荐牛人页返回404 HTML（不可用），需从 `card.innerText` 直接读取公司名判断行业
- **标签结构**：推荐/精选/最新，各15人固定，无翻页/无限滚动
- **打招呼成功弹窗**：顶层document弹出「已向牛人发送招呼」+ 相似候选人推荐面板(`.similar-geek-wrap`)需关闭
- **候选人信息提取**：`card.innerText` 包含姓名/年龄/学历/实习经历等，每人约200字符

---

## 处理新招呼 SOP（沟通页）

### 概述
在沟通页的「新招呼」列表中，对主动投递/系统匹配的候选人按岗位画像进行批量筛选：匹配→求简历+换电话，不匹配→标为不合适。

> **术语说明**：用户所说「打招呼」= 点击「求简历」按钮（非发消息）。报告输出到 `ai-boss\process-hi\` 目录。
> **报告要求**：无论通过与否，所有候选人都必须写进报告（匹配+淘汰+错误全列出）。

### 流程步骤

#### Step 0: 确认岗位画像（必须）
- **启动前必须ask_user确认**：目标岗位名称 + 筛选条件
- 示例问法：「请确认要处理哪个岗位的新招呼？需要的筛选条件是什么？（如：年龄、学历、性别、经验要求）」
- 用户回复后，提取并确认：岗位名、各项筛选条件、匹配/不匹配时的操作

#### Step 1: 导航到新招呼页面
- URL: `https://www.zhipin.com/web/chat/new_greet`
- ⚠️ **SPA重定向**：实际会跳转到 `chat/index`，新招呼是其中的标签页，不是独立页面
- 切换标签：`document.querySelectorAll('.chat-label-item')` 找 title 含「新招呼」→ **必须 mouseenter + click(bubbles:true)** 才生效
- 确认标签激活：检查 `.chat-label-item.selected` 的 title 含「新招呼」

#### Step 2: 选择目标岗位（必须）
- 在新招呼列表顶部，有岗位筛选下拉
- **必须先选择目标岗位**（如「后台开发工程师」），否则列表混杂多岗位候选人
- **已验证选择器**：`.dropmenu-label.chat-select-job` → mouseenter+click → 等500ms → `.ui-dropmenu-list li` 含目标岗位名 → mouseenter+click
- 注意：同样需要 mouseenter 触发hover态后再 click，否则click无响应

#### Step 3: 注入自动化脚本
- 注入后台IIFE脚本，含：
  - 浮动状态面板（右上角，显示进度/统计/停止按钮）
  - `window._stopBossBot = true` 可随时中止
  - `window._bossBotStats` 可查询进度

#### Step 4: 逐人评估与操作

**评估逻辑（从右侧面板 `.base-info-single-container` 提取）：**
- **年龄**：`baseText.match(/(\d+)岁/)`
- **学历**：匹配 `本科|硕士|博士` 等
- **性别**：`baseHTML.includes('icon-icon-man')` = 男；`icon-icon-woman` = 女（⚠️注意单数，非men/women）
- **经验关键词**：搜索 baseText + 聊天消息中的行业/技术关键词

**操作-不合适：**
1. 点击 `span.operate-btn`（文本="不合适"），需 mouseenter + click
2. 等待弹窗出现（`.not-fit-wrap` 内层，height > 0）
3. 点击对应 `.reason-item`（薪资不符/学历不符/年龄不符/期望不符/距离太远/过往经历不符/简历不真实/已找到工作/其他原因）
4. 系统自动跳到下一位候选人

**操作-匹配（求简历+换电话）：**
1. 点击 `span.operate-btn`（文本="求简历"）
2. 确认弹窗中点击确认按钮
3. 点击 `span.operate-btn`（文本="换电话"，需检查是否disabled）
4. 手动点击下一个 `.geek-item` 前进

#### Step 5: 列表滚动加载
- 初始加载约40条，处理后自动前进
- 列表耗尽时，优先滚动 `.chat-user-list` / `.geek-list`；若页面滚动无效，改找包含 `.geek-item` 的真实可滚动祖先并滚动它触发加载

### 关键技术细节（已验证）
- **左侧列表项**：`.geek-item`，选中态 `.geek-item.selected`
- **候选人名**：`.geek-name`；**岗位**：`.source-job`
- **求简历按钮**：`span.operate-btn`（常在 `.operate-icon-item` 下）
- **求简历确认层**：可能是 hidden 的 `.exchange-tooltip`；若“确定”文本节点 rect=0，仍可直接点击 `.boss-btn-primary.boss-btn` 完成确认
- **不合适按钮嵌套**：外层 `.not-fit-wrap`（始终可见）→ 内层弹窗（display:none，点击后显示）
- **操作节奏**：~2s主延迟 + 0.8s短延迟，约5秒/人
- **脚本控制**：`window._stopBossBot=true` 停止；`window._bossBotStats` 读状态

### 补充验证（新招呼场景）
- **基础信息来源**：新招呼页性别/年龄/学历不要只读 DOM；优先从 `window.__chatStore` 读取：`gender`（1男/2女）、`geekInfo.geekTag`（tagType 2=年龄，3=学历）
- **简历经验来源**：金融/Java 经验通过 API `/wapi/zpgeek/resume/coach/resume.json?geekId=XXX&jobId=XXX&security=1` 检查 `experience[]`、`projectExperience[]`、`education[]`
- **评估顺序**：先用 chatStore 做年龄/学历/性别过滤，再调简历 API 判断金融+Java，减少误判
- **Tab/岗位筛选持久性**：Bot 注入前必须再次确认 `.chat-label-item.selected` 和岗位下拉已正确激活；运行中不会自动重选
- **数量含义**：用户给的数量是“处理上限”，不是要求必须匹配到该数量
- **额外岗位校验**：即使 UI 已筛选，仍应检查 `.source-job` 文本，跳过非目标岗位候选人
- **等待详情同步**：点击 `.geek-item` 后，轮询 `window.__chatStore.geekId` 变化（≤3s）再读详情，避免拿到上一个候选人数据
- **脚本注入限制**：此环境 `web_execute_js` 的 `script:` 不支持文件路径，需将脚本代码内联注入
- **⚠️ 物理点击必须用physClick**：直接 `.click()` 在Boss直聘无效，必须依次派发 mouseenter/mouseover/mousedown/mouseup/click 事件序列，否则按钮不响应（血泪教训）
- **不合适原因选项**（已验证）：薪资不符|学历不符|年龄不合适|期望不符|距离太远|过往经历不符|简历不真实|已找到工作|其他原因；性别不符→选「过往经历不符」或「其他原因」
- **求简历禁用(双方回复后可用)**：说明候选人已在沟通中求简历已锁；跳过即可
- **分3段注入脚本**：Part1工具函数→Part2面板→Part3主循环，避免单次超长脚本被截断；**禁止重复注入Part3**，否则多主循环并发导致重复处理候选人超出目标数量，注入前先检查 `window._bossRunning!==true`
- **geekId更新等待**：点击候选人后必须轮询 `window.__chatStore.geekId` 变化再读数据，否则读到旧人数据
- **滚动加载**：页面整体滚动可能无效，需定位包含 `.geek-item` 的真实可滚动祖先触发加载