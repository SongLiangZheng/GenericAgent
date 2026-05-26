# 猎聘网招聘打招呼 SOP (lpt.liepin.com)

## 交互协议（与用户的对话规范）
- **主动打招呼**：用户提供岗位名称 → 我向用户明确「快捷搜索关键词」及「人才画像+数量（可选）」后再执行
- **处理新招呼**：用户提供岗位名称 → 我向用户明确「人才画像+数量（可选）」后再执行（无需确认快捷搜索）

## 关键技术（已验证）
- **搜索入口（最可靠）**: 直接URL导航 `https://lpt.liepin.com/search`
- **搜索框**: `input#searchInput`，需 native setter + `input`/`change` 事件，再 click `div.search-btn`（注意是div非button）
- **快捷标签（已验证正确方法）**: 不要写死 `后端JAVA`；实际可见 `data-title` 可能是 `后端JAVA开发`。优先按可见文本或 `data-title` 模糊匹配，如 `span[role=button][data-title*="后端JAVA"]`。
  ⚠️ **必须先 `el.scrollIntoView({behavior:'instant',block:'center'})` 再派发 mousedown/mouseup/click**，否则标签在视口外（y为负值）点击无效。
  ```javascript
  const tag = document.querySelector('span[role=button][data-title*="后端JAVA"]')
    || [...document.querySelectorAll('span[role=button]')].find(el => /后端JAVA/.test(el.innerText||''));
  tag.scrollIntoView({behavior:'instant', block:'center'});
  await new Promise(r=>setTimeout(r,300));
  const r = tag.getBoundingClientRect();
  ['mousedown','mouseup','click'].forEach(ev=>
    tag.dispatchEvent(new MouseEvent(ev,{bubbles:true,cancelable:true,view:window,
      clientX:r.left+r.width/2, clientY:r.top+r.height/2})));
  // 等待2.5s后页面刷新
  ```
- **候选人卡片**: 选择器 `.resumeCard--nlpbh`，直接读 `innerText`，含姓名/年龄/学历/所在地/行业标签/最近工作/院校
- **院校筛选**: 找筛选区 `input[placeholder*='院校']` 输入院校名 → 无下拉时直接点「确定」按钮也生效（不依赖下拉选项）
- **当前行业筛选（高效）**: 点击「当前行业」旁的 `.ant-lpt-select` 下拉 → 行业弹窗出现 → 点左侧「金融」菜单项 → 右侧子类点「不限」 → 点「确认」，可将33页候选缩减到几张卡片
- **打招呼流程（路径A：超级聊聊）**: 点卡片上 `button.resumeItemSuperChat` → 弹出「请选择开聊职位」模态框 → 确认「后端开发工程师」已选中（`.active--nJIZF`）→ 点「确认」按钮
- **打招呼流程（路径B：立即沟通，已验证）**: 点卡片内容区（触发加载）→ 等预览面板（`.resume-detail-content-body`）出现 → 找预览面板内的「立即沟通」button → 点击 → 弹出职位modal → 选li（`modal.querySelectorAll('li')` 找目标职位）→ 确认（需pointer事件序列，见坑点）
  - ⚠️ **区分**：卡片上的「立即沟通」span=只展开详情；**预览面板内**的「立即沟通」button=真正打招呼
  - ⚠️ **确认按钮坑（关键）**：普通 `.click()` 可能失效，必须带坐标完整事件序列：
    ```javascript
    const rect = btn.getBoundingClientRect(), x=rect.left+rect.width/2, y=rect.top+rect.height/2;
    for(const ev of ['pointerover','pointerenter','mouseover','mouseenter','pointermove','mousemove','pointerdown','mousedown','pointerup','mouseup','click']) {
      btn.dispatchEvent(new (ev.startsWith('pointer')?PointerEvent:MouseEvent)(ev,{bubbles:true,cancelable:true,view:window,clientX:x,clientY:y,pointerId:1,pointerType:'mouse'}));
      await new Promise(r=>setTimeout(r,30));
    }
    ```
- **成功判断**: 点确认后 transient 出现招呼语（如「我对你的履历非常感兴趣」）+ 「发送成功」即为成功

## 筛选逻辑（推荐顺序）
1. **院校筛选**：筛选区输入目标院校名 → 点「确定」（无需下拉选项）
2. **当前行业筛选（高效！）**：点「当前行业」下拉 → 选「金融」→「不限」→ 点确认，可将结果从几十页缩减到一页

## 典型坑
- **搜索结果为空**：等待 3s 再 scan；或直接 URL 带参数导航绕过JS输入问题
- **首次搜索为空**：可先点快捷标签（`.tag-item` 如 `java开发`）激活页面，再重新搜索
- **院校筛选无匹配**：不依赖筛选器，改为逐卡片 `innerText` 读院校字段手动判断
- **成功判断**：确认后可能无成功弹窗，但打招呼实际已成功。验证方式：检查 `button.xpath-open-im-btn` 文字是否变为「继续沟通」
- **IM窗口自动弹出**：打招呼成功后可能自动打开IM聊天窗口，关闭用 `.im-ui-basic-chat-header-modal-close`
- **翻页**：`.ant-lpt-pagination-item[title="N"]` 点击翻页

---

## 新招呼处理流程（已验证）

### 入口
- 导航：猎聘沟通页 `https://lpt.liepin.com/e/index`
- 左侧 IM 筛选器点击「新招呼」tab（`span[role=tab]` 文字匹配「新招呼」）
- **岗位筛选下拉框（已验证 v2，可靠方法）**：
  ⚠ **直接 `.click()` 经常不打开 dropdown**（select 变 focused 但无 dropdown DOM）。必须用完整事件链+键盘事件触发：
  ```javascript
  // 1. 完整事件链打开 dropdown
  const selector = document.querySelector('.im-ui-im-job-filter .ant-im-select-selector');
  const input = document.querySelector('#rc_select_0');
  for(const ev of ['mousedown','mouseup','click']) {
    selector.dispatchEvent(new MouseEvent(ev, {bubbles:true, cancelable:true, view:window}));
  }
  input.focus();
  await new Promise(r=>setTimeout(r,400));
  input.dispatchEvent(new KeyboardEvent('keydown', {key:'ArrowDown', keyCode:40, bubbles:true}));
  await new Promise(r=>setTimeout(r,600));
  
  // 2. 找到目标 option（option 元素含子节点job描述，不可用 leaf 节点过滤！）
  const dd = document.querySelector('.ant-im-select-dropdown');
  const target = [...dd.querySelectorAll('*')]
    .filter(el => el.offsetHeight > 0)
    .find(el => (el.innerText||'').trim().startsWith('WEB前端开发'));  // 替换岗位名
  
  // 3. dispatchEvent 完整链点击 option（防止纯 .click() 失效）
  const rect = target.getBoundingClientRect();
  for(const ev of ['mousedown','mouseup','click']) {
    target.dispatchEvent(new MouseEvent(ev, {bubbles:true,cancelable:true,clientX:rect.x+rect.width/2,clientY:rect.y+rect.height/2}));
  }
  await new Promise(r=>setTimeout(r,800));
  // 验证: document.querySelector('.ant-im-select-selection-item').innerText === '目标岗位名'
  ```
  **坑点记录**：
  - 单纯 `selector.click()` 失败率高，必须 mousedown+mouseup+click
  - input.focus() + ArrowDown 键盘事件是触发 rc-select 打开的关键
  - option 元素有子节点（岗位描述），不能用 `children.length === 0` 过滤
  - option 类名 `ant-im-select-item ant-im-select-item-option im-ui-recruit-d`，用文本匹配最稳

### 前置确认（重要！）
1. **岗位画像确认**：明确岗位后，**必须与用户确认必备资质**，并将资质存入本 SOP `## 岗位画像` 章节，后续无需重复确认
2. **处理数量**：用户可指定处理数量 N（默认全部处理）

### JS 批量处理脚本逻辑（window 全局状态）
```javascript
window.liepin_running = true;  // 可设 false 随时停止
window.liepin_results = [];    // 每条结果存入
window.liepin_status = '';     // 进度日志
```
**核心流程**：
1. 读取当前已选中候选人的 profile panel 信息（年龄、工作经历文本）
2. 评估是否满足资质（age < N, 金融关键词等）
3. 点击「不合适」或「索要简历」（`.im-ui-chat-content-wrapper` 内 span 文字匹配）
4. **sleep 1000ms**（等待自动跳转下一个），再处理下一个
5. **不主动切换候选人**：「不合适」后页面自动加载下一个；「索要简历/看简历」后也 sleep 后再重新获取当前 profile

**⚠ 关键规则**：
- **一个处理完再到下一个，中间 sleep 1s**，禁止在两个招呼间来回切换
- 「不合适」点击后联系人自动从列表移除并加载下一个（约 500~1000ms）
- 已有简历（有「看简历」按钮）→ 发常用语第一条（点聊天窗口底部第二个图标 → 选第一条）
- 无简历 → 点「索要简历」

### 关键选择器
- 候选人 profile 面板：`.im-ui-chat-content-wrapper`（innerText 读年龄/工作经历）
- 操作按钮（已验证，两种情况）：
  - `不合适`：`.chatwin-action .actions-right .im-ui-action-button`（优先）；或全局 `[...document.querySelectorAll('span')].find(el=>el.children.length===0&&el.offsetHeight>0&&el.innerText?.trim()==='不合适'&&el.closest('.ant-im-btn,.chatwin-action'))`
  - `索要简历`：`.im-ui-action-button.action-item.action-resume`
  - `查看简历`：`.im-ui-pro-chat-header-basic-info-operate button`
  - ⚠ `.im-ui-chat-content-wrapper` 内直接找 span 不可靠，用上述具体容器选择器更稳定
- 金融关键词：`金融|银行|证券|基金|保险|期货|信托|理财|赢时胜|恒生|金证|东方财富|同花顺|资管|券商|资产管理|财富管理|交易所|清算|结算|核心系统`

### 停止机制
```javascript
window.liepin_running = false;  // 注入此行即可停止
// 读取结果：window.liepin_results, window.liepin_status
```

### 工作报告
- 主动打招呼和新招呼处理，完成后均在 `ai-hire/new-hi/` 目录生成 HTML 工作报告
- 文件名：`liepin_report_YYYYMMDD_HHMMSS.html`
- 内容：岗位、处理时间、总计、合适/不合适人数、候选人明细表（姓名/年龄/金融经验/操作/原因）
- 风格：统一用白底卡片、绿色合适/红色不合适徽标，表格展示明细

---

## 岗位画像（已确认存档）

### WEB前端开发工程师
**必备资质**：
1. 性别男
2. 30岁以下
3. 熟悉Vue3并有相关工作经验（工作经历/项目经历中出现Vue3相关内容，非仅技能标签）

**Vue3经验判断关键词**（工作经历/项目经历中出现任一）：
`Vue3、Vue 3、vue3、Composition API、Vite`

**性别判断**：profile panel innerText 中查找「男」/「女」字（通常在年龄附近，如「25岁 男」）

---

### 后端开发工程师
**必备资质**：
1. 30岁以下
2. 有金融开发经验（曾在金融科技/银行/证券/基金/保险等机构做后端开发，或项目涉及金融核心系统/交易系统/资产管理等）

**金融经验判断关键词**（工作经历/项目经历中出现任一）：
`金融、银行、证券、基金、保险、期货、信托、理财、资管、券商、赢时胜、恒生、金证、东方财富、同花顺、资产管理、财富管理、交易所、交易系统、清算、结算、核心系统、支付清算`