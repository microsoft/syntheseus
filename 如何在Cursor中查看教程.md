# 🎯 如何在Cursor中查看Syntheseus教程

## 📍 文件位置

所有教程文件都在：**`/workspace`** 目录

这是你在Cursor中打开的项目根目录！

---

## 🔍 3种方法在Cursor中找到文件

### ⭐ 方法1：文件浏览器（推荐）

1. **点击左侧的文件图标**（或按 `Cmd+Shift+E` / `Ctrl+Shift+E`）
2. **向下滚动**，你会看到以下文件：

```
/workspace
├── 多步逆合成教程-README.md        ← 从这里开始！
├── 教程文件索引.md                  ← 文件导航
├── 简单示例.py                      ← 可运行示例
├── 如何在Cursor中查看教程.md        ← 你正在看的这个
└── docs/
    ├── 多步逆合成使用指南.md        ← 详细文档
    └── tutorials/
        └── 多步逆合成完整教程.ipynb ← Notebook教程
```

3. **双击任意文件即可打开**

**提示**：如果看不到文件，在文件浏览器空白处**右键 → 刷新**

---

### ⚡ 方法2：快速打开（最快）

1. 按 **`Cmd+P`** (Mac) 或 **`Ctrl+P`** (Windows/Linux)
2. 输入文件名的一部分：
   - 输入 `教程` 或 `README`  
   - 输入 `示例` 或 `simple`
   - 输入 `notebook` 或 `ipynb`
3. 用方向键选择，按回车打开

---

### 💻 方法3：使用终端

1. 在Cursor中打开终端：
   - 按 **`` Ctrl+` ``** (反引号)
   - 或 菜单栏：**终端 → 新建终端**

2. 运行以下命令：

```bash
# 查看所有教程文件
ls -lh /workspace/*教程* /workspace/*示例*

# 快速阅读入门指南
cat /workspace/多步逆合成教程-README.md

# 查看文件索引
cat /workspace/教程文件索引.md

# 运行Python示例
python /workspace/简单示例.py
```

---

## 📚 推荐的学习顺序

### 🚀 快速开始（5分钟）

1. **打开**：`多步逆合成教程-README.md`
   - 快捷键：`Cmd/Ctrl+P` → 输入 `README`
2. **阅读**：快速了解功能
3. **运行**：在终端执行 `python /workspace/简单示例.py`

### 📖 深入学习（2小时）

1. **打开**：`docs/tutorials/多步逆合成完整教程.ipynb`
   - 需要Jupyter：`pip install jupyter`
   - 运行：`jupyter notebook`
2. **学习**：逐个单元格运行代码
3. **实验**：修改参数观察效果

### 📚 完整参考（需要时查阅）

1. **打开**：`docs/多步逆合成使用指南.md`
2. **用途**：遇到问题时查找解决方案
3. **内容**：包含所有API和最佳实践

---

## ❓ 常见问题

### Q: 我看不到中文文件名？

**A**: 文件名是中文的，确保：
- Cursor设置中字体支持中文
- 或者在终端用 `ls` 命令查看
- 或使用快速打开（`Cmd/Ctrl+P`）搜索

### Q: 文件在哪个目录？

**A**: 所有文件都在 `/workspace` 目录，这就是：
- 你在Cursor中打开的项目根目录
- 终端中的当前目录（运行 `pwd` 确认）
- 文件浏览器显示的顶层目录

### Q: 如何打开Jupyter Notebook？

**A**: 
```bash
# 方法1：在终端运行
cd /workspace
jupyter notebook docs/tutorials/多步逆合成完整教程.ipynb

# 方法2：使用JupyterLab
jupyter lab docs/tutorials/多步逆合成完整教程.ipynb

# 方法3：在Cursor中直接打开.ipynb文件
# Cursor可能会显示JSON格式，此时需要用浏览器打开
```

### Q: 如何运行Python示例？

**A**:
```bash
# 确保在正确目录
cd /workspace

# 运行示例
python 简单示例.py

# 或使用完整路径
python /workspace/简单示例.py
```

---

## 🎨 Cursor特定提示

### 使用AI助手查看文件

在Cursor的聊天窗口中输入：
```
@多步逆合成教程-README.md 总结一下这个教程
```

### 使用Composer编辑

1. 按 `Cmd+I` / `Ctrl+I` 打开Composer
2. 选中文件或代码
3. 让AI帮你修改或解释

### 使用Chat询问

在右侧Chat面板中：
```
这个项目的教程在哪里？
如何运行多步逆合成示例？
```

---

## ✅ 验证文件存在

在终端运行以下命令验证所有文件：

```bash
echo "检查教程文件..."
test -f /workspace/多步逆合成教程-README.md && echo "✅ README存在"
test -f /workspace/简单示例.py && echo "✅ Python示例存在"
test -f /workspace/docs/多步逆合成使用指南.md && echo "✅ 使用指南存在"
test -f /workspace/docs/tutorials/多步逆合成完整教程.ipynb && echo "✅ Notebook存在"
test -f /workspace/教程文件索引.md && echo "✅ 文件索引存在"
echo "检查完成！"
```

---

## 🚀 现在就开始！

### 最快的开始方式：

1. 按 **`Cmd/Ctrl+P`**
2. 输入 **`README`**
3. 打开 **`多步逆合成教程-README.md`**
4. 开始阅读！

或者在终端直接运行示例：

```bash
python /workspace/简单示例.py
```

---

**祝学习愉快！** 🎉

如果还有问题，在Cursor的Chat中问我："教程文件在哪里？"
