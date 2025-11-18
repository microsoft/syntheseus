# Syntheseus 多步逆合成完整教程

欢迎使用 Syntheseus 进行多步逆合成规划！本教程提供了完整的中文指南，帮助你快速掌握如何控制原料库、单步算法、搜索算法并进行评测。

## 📚 教程内容

本教程包含两个主要文件：

### 1. 📓 交互式 Jupyter Notebook
**文件位置**: `docs/tutorials/多步逆合成完整教程.ipynb`

这是一个完整的、可运行的 Jupyter Notebook，包含：

- ✅ **环境准备**: 导入必要的库和模块
- ✅ **单步反应模型**: 7种主流模型的使用和比较
- ✅ **原料库配置**: 从列表或文件创建和管理原料库
- ✅ **搜索算法**: Retro*, MCTS, BFS 等算法的详细配置
- ✅ **运行搜索**: 完整的搜索执行流程
- ✅ **结果评测**: 路径提取、统计分析、性能比较
- ✅ **高级功能**: 批量处理、自定义函数、结果保存
- ✅ **可视化**: 合成路径的图形化展示

**使用方法**:
```bash
cd /workspace
jupyter notebook docs/tutorials/多步逆合成完整教程.ipynb
```

### 2. 📖 详细文档指南
**文件位置**: `docs/多步逆合成使用指南.md`

这是一份全面的 Markdown 文档，包含：

- 📋 **完整目录**: 清晰的章节结构
- 🔍 **核心组件详解**: 单步模型、原料库、搜索算法
- 💡 **最佳实践**: 算法选择、参数调优
- 🚀 **高级用法**: 自定义组件、外部集成
- 🛠 **命令行工具**: 生产环境使用指南
- ❓ **常见问题**: 问题排查和解决方案
- 📊 **性能比较**: 各算法和模型的对比

## 🎯 快速开始

### 最简单的示例（5分钟）

```python
from syntheseus import Molecule
from syntheseus.reaction_prediction.inference import LocalRetroModel
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.algorithms.breadth_first import AndOr_BreadthFirstSearch

# 1. 初始化模型
model = LocalRetroModel(use_cache=True, default_num_results=50)

# 2. 设置原料库
inventory = SmilesListInventory(smiles_list=[
    "Cc1ccc(B(O)O)cc1",
    "Cc1ccc(Br)cc1",
])

# 3. 配置搜索
search = AndOr_BreadthFirstSearch(
    reaction_model=model,
    mol_inventory=inventory,
    limit_iterations=100,
    time_limit_s=60.0
)

# 4. 运行搜索
target = Molecule("Cc1ccc(-c2ccc(C)cc2)cc1")
graph, _ = search.run_from_mol(target)

# 5. 提取路径
from syntheseus.search.analysis.route_extraction import iter_routes_time_order
routes = list(iter_routes_time_order(graph, max_routes=10))
print(f"找到 {len(routes)} 条合成路径")
```

## 📦 核心功能一览

### 1️⃣ 单步反应模型 (7种模型)

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **LocalRetro** | 基于模板，快速 | 通用推荐 |
| **Chemformer** | Transformer | 新颖反应 |
| **MEGAN** | 图编辑 | 平衡性能 |
| **MHNreact** | 超图网络 | 复杂反应 |
| **Graph2Edits** | 图编辑 | 灵活 |
| **RetroKNN** | 基于检索 | 可解释性 |
| **RootAligned** | 根对齐 | 通用 |

**使用示例**:
```python
from syntheseus.reaction_prediction.inference import LocalRetroModel

model = LocalRetroModel()
[results] = model([Molecule("Cc1ccc(-c2ccc(C)cc2)cc1")], num_results=10)
```

### 2️⃣ 原料库管理

**从 SMILES 列表创建**:
```python
from syntheseus.search.mol_inventory import SmilesListInventory

inventory = SmilesListInventory(
    smiles_list=["Cc1ccc(Br)cc1", "Ic1ccccc1"],
    canonicalize=True
)
```

**从文件加载**:
```python
inventory = SmilesListInventory.load_from_file("building_blocks.txt")
```

### 3️⃣ 搜索算法 (3种主流算法)

#### Retro* - 寻找最优路径
```python
from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator

alg = RetroStarSearch(
    reaction_model=model,
    mol_inventory=inventory,
    value_function=ConstantNodeEvaluator(0.0),
    limit_iterations=100,
    time_limit_s=60.0,
)
```

#### MCTS - 探索多样路径
```python
from syntheseus.search.algorithms.mcts.molset import MolSetMCTS
from syntheseus.search.node_evaluation.common import HasSolutionValueFunction

alg = MolSetMCTS(
    reaction_model=model,
    mol_inventory=inventory,
    reward_function=HasSolutionValueFunction(),
    limit_iterations=100,
    time_limit_s=60.0,
)
```

#### BFS - 简单基线
```python
from syntheseus.search.algorithms.breadth_first import AndOr_BreadthFirstSearch

alg = AndOr_BreadthFirstSearch(
    reaction_model=model,
    mol_inventory=inventory,
    limit_iterations=100,
)
```

### 4️⃣ 评测与分析

**提取路径**:
```python
from syntheseus.search.analysis.route_extraction import iter_routes_time_order

routes = list(iter_routes_time_order(graph, max_routes=10))
```

**计算求解时间**:
```python
from syntheseus.search.analysis.solution_time import get_first_solution_time

soln_time = get_first_solution_time(graph)
```

**可视化**:
```python
from syntheseus.search.visualization import visualize_andor

visualize_andor(graph, filename="route.pdf", nodes=routes[0])
```

## 🔧 命令行工具

对于生产环境，可以使用命令行接口：

```bash
# 运行单个目标搜索
python -m syntheseus.cli.search \
    search_target="Cc1ccc(-c2ccc(C)cc2)cc1" \
    inventory_smiles_file=building_blocks.txt \
    model_class=LocalRetro \
    search_algorithm=retro_star \
    time_limit_s=60 \
    results_dir=./results

# 批量搜索
python -m syntheseus.cli.search \
    search_targets_file=targets.txt \
    inventory_smiles_file=building_blocks.txt \
    model_class=LocalRetro \
    search_algorithm=retro_star \
    results_dir=./batch_results
```

## 📊 算法选择指南

| 需求 | 推荐算法 | 原因 |
|------|---------|------|
| 最优路径 | **Retro*** | 理论保证最优 |
| 多样路径 | **MCTS** | 探索能力强 |
| 快速原型 | **BFS** | 简单稳定 |
| 大规模搜索 | **Retro*** | 效率最高 |

## 🎓 学习路径

1. **入门** (30分钟)
   - 阅读快速开始部分
   - 运行简单示例
   - 理解基本概念

2. **进阶** (2小时)
   - 学习 Jupyter Notebook
   - 尝试不同模型和算法
   - 理解参数影响

3. **高级** (1天)
   - 阅读完整文档
   - 实现自定义组件
   - 在自己的数据上测试

4. **专家** (持续)
   - 优化搜索策略
   - 集成到生产环境
   - 贡献代码和算法

## 💡 最佳实践

### 性能优化

1. **启用缓存**: `use_cache=True` - 避免重复计算
2. **GPU加速**: `use_gpu=True` - 提升模型速度
3. **限制深度**: `max_expansion_depth=10` - 避免过深搜索
4. **批处理**: 合理设置 `batch_size` - 平衡速度和内存

### 结果质量

1. **更大原料库**: 使用商业级building blocks
2. **更多模型调用**: 增加 `limit_reaction_model_calls`
3. **组合多个模型**: 集成多个单步模型的预测
4. **调整搜索参数**: 根据任务特点优化

### 开发建议

1. **先用BFS调试**: 简单算法便于问题定位
2. **小规模测试**: 在小数据集上验证流程
3. **保存中间结果**: 便于分析和调试
4. **版本控制**: 记录参数配置和结果

## 📝 常见问题

<details>
<summary><b>Q: 找不到合成路径怎么办？</b></summary>

A: 
1. 扩大原料库
2. 增加搜索时间和模型调用次数
3. 尝试MCTS算法
4. 检查SMILES格式是否正确
</details>

<details>
<summary><b>Q: 如何提高搜索速度？</b></summary>

A:
1. 启用缓存: `use_cache=True`
2. 使用GPU: `use_gpu=True`
3. 限制搜索深度和迭代次数
4. 减少返回结果数
</details>

<details>
<summary><b>Q: 内存不足怎么办？</b></summary>

A:
1. 减小批处理大小
2. 限制图大小: `limit_graph_nodes`
3. 不保存完整图: `save_graph=false`
4. 使用更小的模型
</details>

## 🔗 相关资源

### 官方资源
- 📖 [完整文档](https://microsoft.github.io/syntheseus/)
- 💻 [GitHub](https://github.com/microsoft/syntheseus)
- 📝 [论文](https://pubs.rsc.org/en/content/articlelanding/2024/fd/d4fd00093e)

### 社区资源
- 💬 [Issue跟踪](https://github.com/microsoft/syntheseus/issues)
- 🎓 [示例代码](https://github.com/microsoft/syntheseus/tree/main/docs/tutorials)

### 相关项目
- [Retro-fallback](https://github.com/AustinT/retro-fallback-iclr24)
- [RetroGFN](https://github.com/gmum/RetroGFN)
- [SimpRetro](https://github.com/catalystforyou/SimpRetro)

## 📄 引用

如果你在研究中使用了 Syntheseus，请引用：

```bibtex
@article{maziarz2024re,
  title={Re-evaluating retrosynthesis algorithms with syntheseus},
  author={Maziarz, Krzysztof and Tripp, Austin and Liu, Guoqing and Stanley, Megan and Xie, Shufang and Gainski, Piotr and Seidl, Philipp and Segler, Marwin},
  journal={Faraday Discussions},
  year={2024},
  publisher={Royal Society of Chemistry}
}
```

## 🤝 贡献

欢迎贡献代码、文档或报告问题！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📜 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

**开始你的逆合成规划之旅吧！** 🚀

有任何问题欢迎在 GitHub 上提 Issue！
