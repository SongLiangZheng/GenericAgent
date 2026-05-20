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
- **候选人卡片**：`.candidate-card-wrap`（iDoc内查询）
- **打招呼按钮**：`.btn.btn-greet`；打招呼后变为 `.btn-continue`（继续沟通）= 成功标志
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
- **性别**：`baseHTML.includes('icon-icon-men')` = 男；`icon-icon-women` = 女
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
- 列表耗尽时，滚动 `.chat-user-list` / `.geek-list` 触发加载更多

### 关键技术细节（已验证）
- **左侧列表项**：`.geek-item`，选中态 `.geek-item.selected`
- **候选人名**：`.geek-name`；**岗位**：`.source-job`
- **不合适按钮嵌套**：外层 `.not-fit-wrap`（始终可见）→ 内层弹窗（display:none，点击后显示）
- **操作节奏**：~2s主延迟 + 0.8s短延迟，约5秒/人
- **脚本控制**：`window._stopBossBot=true` 停止；`window._bossBotStats` 读状态