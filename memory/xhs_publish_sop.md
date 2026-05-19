# XHS 小红书发布 SOP

## 图片生成（防重叠）
- 工具: `D:\workspace\me\xhs\card_generator.py`
- `CardGenerator(output_dir).make_cover(title, subtitle, palette_idx=0)` / `.make_content_card(index, total, badge, title, sections, summary, palette_idx=0)` — **palette_idx在方法参数上，不在构造器**
- 坑: md正文用field()提取时终止符须写具体块名，如`(?=\n\*\*(?:封面标题|卡片提纲|正文|标签)\*\*)`，否则遇正文内粗体即截断
- 核心原理: 用 `textbbox()` 精确测量文字高度，`auto_shrink_font()` 超长自动缩字，`draw_text_block()` 返回实际底部Y坐标
- 验证: `validate_card(path)` / `validate_all(dir)` 用 RapidOCR 打分，score>=90为通过
- 坑: 同行OCR误判(同行文字被分为2个bbox)→用Y中心差过滤; print中勿用emoji(GBK控制台报错)

## 关键前置
- MCP publish_content **不能用**（独立浏览器实例，无用户登录态）
- 必须直接操作已登录的用户浏览器（TMWebDriver）
- 发布URL: https://creator.xiaohongshu.com/publish/publish

## 补充坑点（2026-05-08验证）
- feeds文件首行可能为`FAILED: curl`，无法直接JSON解析，需用正则按displayTitle匹配修改used字段
- `file_patch`必须精确匹配`old_content`的换行、缩进，建议先用`file_read`获取准确原文后再操作
- 导入`card_generator`需先执行`sys.path.insert(0, r'D:\workspace\me\xhs')`，否则报ModuleNotFoundError
- `validate_all`验证图片时需确保`overlaps=0`且`score=100`才通过
- 发布图片上传必须用CDP batch（同批次获取nodeId），分步操作会导致nodeId过期
- 正文标签需逐个插入，每次插入后等待300ms再按Enter确认

## 发布流程

### 1. 导航到发布页并切换到「上传图文」
```js
// 确认在发布页，点击"上传图文" tab
const tab = Array.from(document.querySelectorAll('.creator-tab')).find(t => t.textContent.includes('上传图文'));
tab.click();
```

### 2. 上传图片（CDP batch 3步合一 - nodeId必须同batch内fresh获取）
```json
{"cmd": "batch", "commands": [
  {"cmd": "cdp", "tabId": TAB_ID, "method": "DOM.getDocument", "params": {"depth": 1}},
  {"cmd": "cdp", "tabId": TAB_ID, "method": "DOM.querySelector", "params": {"nodeId": "$0.root.nodeId", "selector": "input[type='file']"}},
  {"cmd": "cdp", "tabId": TAB_ID, "method": "DOM.setFileInputFiles", "params": {"nodeId": "$1.nodeId", "files": ["绝对路径1", "绝对路径2"]}}
]}
```
⚠️ 不能分步（nodeId会过期），必须同一batch

### 3. 填写标题
```js
const input = document.querySelector('input.d-text[placeholder="填写标题会有更多赞哦"]');
const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
setter.call(input, '标题内容');
input.dispatchEvent(new Event('input', {bubbles: true}));
```

### 4. 填写正文（ProseMirror编辑器）
```js
const editor = document.querySelector('.tiptap.ProseMirror');
editor.focus();
document.execCommand('selectAll');
document.execCommand('insertText', false, '正文内容（含\n换行）');
```
- ⚠️ 话题识别：正文里输入 `#话题名` 后，编辑器可能先渲染成候选态（如 `span.suggestion`），**按 Enter** 可确认并转成正式话题节点：`<a class="tiptap-topic">#话题名</a>`；仅点击候选列表不一定生效
- ⚠️ 多标签识别：一次性插入所有标签只有最后一个会被识别。正确做法：**逐个插入**，每次 `insertText('#话题名')` 后等待 suggestion 出现（约300ms），再 `dispatchEvent(Enter keydown/keypress/keyup)`，确认后再插入下一个

### 5. 点击发布
```js
const btn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === '发布');
btn.click();
// 成功会出现 .success-page 元素，内含"发布成功"文字
```

## 内容生成流程（用户要求）
0. **话题来源判断**（必须先做）：
   - 读取 `D:\workspace\me\xhs\xiaohongshu-mcp\workplace_feeds.json`，检查是否存在 `used!=true` 的话题
   - 注意: feeds文件格式异常(首行为`FAILED: curl`)，勿用json.loads直接解析，用regex扫描`"used": true/false`行
   - 若有未使用话题 → 直接跳到步骤2，**跳过**市场调研
   - 若全部已用完 → 先对 feeds 文件做滚动备份（重命名为 workplace_feeds_YYYYMMDD.json），再执行市场调研（输出格式与现有 feeds 保持一致，补充新话题），然后继续步骤2
1. 市场热点调研（仅在所有话题已用完时执行，见步骤0）
2. 生成内容（标题≤20字，每篇必须包含：封面标题 + 卡片提纲 + 正文 + 标签，并默认生成对应图片卡片，不能只输出正文）
3. 评估打分（对比爆款最佳实践）
4. 生成图片（封面+3张卡片，neo-brutalism风格，PIL+MicrosoftYaHei字体）
5. 发布

## 图片生成
- 工具: PIL (pillow)
- 字体: `C:/Windows/Fonts/msyh.ttc` (Microsoft YaHei)
- 风格: neo-brutalism（高饱和配色+粗黑边框）
- 尺寸: 1242×1660 (小红书标准)
- 文件: D:\workspace\me\xhs\_workspace\cover.png, card_1.png ...