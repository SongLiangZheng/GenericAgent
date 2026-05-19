# 猎聘网招聘打招呼 SOP (lpt.liepin.com)

## 关键技术（已验证）
- **搜索入口（最可靠）**: 直接URL导航 `https://lpt.liepin.com/search`
- **搜索框**: `input#searchInput`，需 native setter + `input`/`change` 事件，再 click `div.search-btn`（注意是div非button）
- **快捷标签**: `span.tagTitle[data-title="后端JAVA"][role=button]`（非`.tag-item`），点击后搜索栏class变为`.submited`即生效
- **候选人卡片**: 选择器 `.resumeCard--nlpbh`，直接读 `innerText`，含姓名/年龄/学历/所在地/行业标签/最近工作/院校
- **院校筛选**: 找筛选区 `input[placeholder*='院校']` 输入院校名 → 无下拉时直接点「确定」按钮也生效（不依赖下拉选项）
- **当前行业筛选（高效）**: 点击「当前行业」旁的 `.ant-lpt-select` 下拉 → 行业弹窗出现 → 点左侧「金融」菜单项 → 右侧子类点「不限」 → 点「确认」，可将33页候选缩减到几张卡片
- **打招呼流程**: 点卡片上 `button.resumeItemSuperChat`（= 超级聊聊按钮，是真正的打招呼按钮）→ 弹出「请选择开聊职位」模态框 → 确认「后端开发工程师」已选中（`.active--nJIZF`）→ 点「确认」按钮
- **成功判断**: 点确认后 transient 出现招呼语（如「我对你的履历非常感兴趣」）即为成功；也可能无成功弹窗但实际已发送
- **⚠ 卡片上的「立即沟通」span 只是展开详情，不是真正的打招呼按钮**

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