#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Syntheseus 多步逆合成简单示例

这个脚本演示了如何使用 Syntheseus 进行基本的多步逆合成规划。
包括：单步模型初始化、原料库配置、搜索算法运行、结果分析。

使用方法:
    python 简单示例.py
"""

from syntheseus import Molecule
from syntheseus.reaction_prediction.inference import LocalRetroModel
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.algorithms.breadth_first import AndOr_BreadthFirstSearch
from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
from syntheseus.search.algorithms.best_first import retro_star
from syntheseus.search.node_evaluation.common import (
    ConstantNodeEvaluator,
    ReactionModelLogProbCost
)
from syntheseus.search.analysis.route_extraction import iter_routes_time_order
from syntheseus.search.graph.and_or import AndNode
import time


def main():
    print("=" * 60)
    print("Syntheseus 多步逆合成简单示例")
    print("=" * 60)
    
    # ============================================================
    # 1. 定义目标分子
    # ============================================================
    print("\n[1/5] 定义目标分子...")
    target_smiles = "Cc1ccc(-c2ccc(C)cc2)cc1"  # 4,4'-二甲基联苯
    target = Molecule(target_smiles)
    print(f"  目标分子: {target.smiles}")
    
    # ============================================================
    # 2. 初始化单步反应模型
    # ============================================================
    print("\n[2/5] 初始化单步反应模型...")
    print("  使用 LocalRetro 模型（如果是首次运行，会自动下载模型）")
    
    model = LocalRetroModel(
        use_cache=True,           # 启用缓存提高性能
        default_num_results=50    # 每次预测返回50个反应
    )
    print(f"  ✓ 模型加载完成，缓存路径: {model.model_dir}")
    
    # ============================================================
    # 3. 配置原料库
    # ============================================================
    print("\n[3/5] 配置原料库...")
    
    # 创建简单的原料库（实际应用中应使用更大的商业原料库）
    building_blocks = [
        "Cc1ccc(B(O)O)cc1",    # 4-甲基苯硼酸
        "Cc1ccc(Br)cc1",       # 4-溴甲苯
        "Cc1ccc(I)cc1",        # 4-碘甲苯
        "Brc1ccccc1",          # 溴苯
        "Ic1ccccc1",           # 碘苯
        "B(O)(O)c1ccccc1",     # 苯硼酸
        "O=Cc1ccccc1",         # 苯甲醛
        "Cc1ccccc1",           # 甲苯
    ]
    
    inventory = SmilesListInventory(
        smiles_list=building_blocks,
        canonicalize=True  # 自动规范化SMILES
    )
    
    print(f"  ✓ 原料库包含 {len(inventory)} 个建构块")
    print(f"  ✓ 目标分子在原料库中: {inventory.is_purchasable(target)}")
    
    # ============================================================
    # 4. 配置并运行搜索算法
    # ============================================================
    print("\n[4/5] 配置并运行搜索算法...")
    print("  提示: 我们将比较两种算法的性能\n")
    
    results = {}
    
    # -------------------- BFS算法 --------------------
    print("  >> 运行广度优先搜索 (BFS)...")
    bfs_alg = AndOr_BreadthFirstSearch(
        reaction_model=model,
        mol_inventory=inventory,
        limit_iterations=100,
        limit_reaction_model_calls=100,
        time_limit_s=60.0,
    )
    
    start_time = time.time()
    bfs_alg.reset()
    bfs_graph, _ = bfs_alg.run_from_mol(target)
    bfs_time = time.time() - start_time
    
    bfs_routes = list(iter_routes_time_order(bfs_graph, max_routes=10))
    
    results['BFS'] = {
        'time': bfs_time,
        'nodes': len(bfs_graph),
        'routes': len(bfs_routes),
        'calls': model.num_calls()
    }
    
    print(f"     ✓ 完成！用时 {bfs_time:.2f}秒")
    print(f"       探索节点数: {len(bfs_graph)}")
    print(f"       找到路径数: {len(bfs_routes)}")
    print(f"       模型调用次数: {model.num_calls()}\n")
    
    # -------------------- Retro*算法 --------------------
    print("  >> 运行 Retro* 算法...")
    
    # 重新初始化模型以公平比较
    model = LocalRetroModel(use_cache=True, default_num_results=50)
    
    retro_star_alg = RetroStarSearch(
        reaction_model=model,
        mol_inventory=inventory,
        or_node_cost_fn=retro_star.MolIsPurchasableCost(),
        and_node_cost_fn=ReactionModelLogProbCost(normalize=False),
        value_function=ConstantNodeEvaluator(0.0),  # Retro*-0
        limit_iterations=100,
        limit_reaction_model_calls=100,
        time_limit_s=60.0,
        max_expansion_depth=10,
    )
    
    start_time = time.time()
    retro_star_alg.reset()
    retro_star_graph, _ = retro_star_alg.run_from_mol(target)
    retro_star_time = time.time() - start_time
    
    retro_star_routes = list(iter_routes_time_order(retro_star_graph, max_routes=10))
    
    results['Retro*'] = {
        'time': retro_star_time,
        'nodes': len(retro_star_graph),
        'routes': len(retro_star_routes),
        'calls': model.num_calls()
    }
    
    print(f"     ✓ 完成！用时 {retro_star_time:.2f}秒")
    print(f"       探索节点数: {len(retro_star_graph)}")
    print(f"       找到路径数: {len(retro_star_routes)}")
    print(f"       模型调用次数: {model.num_calls()}\n")
    
    # ============================================================
    # 5. 分析和比较结果
    # ============================================================
    print("[5/5] 分析结果...")
    
    # 打印对比表格
    print("\n  算法性能对比:")
    print("  " + "-" * 58)
    print(f"  {'算法':<10} {'时间(秒)':<12} {'节点数':<10} {'路径数':<10} {'模型调用':<10}")
    print("  " + "-" * 58)
    for alg_name, stats in results.items():
        print(f"  {alg_name:<10} {stats['time']:<12.2f} {stats['nodes']:<10} "
              f"{stats['routes']:<10} {stats['calls']:<10}")
    print("  " + "-" * 58)
    
    # 分析最佳路径
    if retro_star_routes:
        print("\n  Retro* 找到的最佳路径:")
        best_route = retro_star_routes[0]
        num_steps = len([n for n in best_route if isinstance(n, AndNode)])
        print(f"    - 反应步数: {num_steps}")
        
        # 打印反应步骤
        print(f"    - 反应序列:")
        for i, node in enumerate([n for n in best_route if isinstance(n, AndNode)], 1):
            rxn = node.reaction
            reactants = " + ".join([r.smiles for r in rxn.reactants])
            print(f"      步骤 {i}: {reactants}")
            print(f"              ↓")
            print(f"              {rxn.product.smiles}")
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("搜索完成！")
    print("=" * 60)
    print("\n下一步建议:")
    print("  1. 尝试更大的原料库以找到更多路径")
    print("  2. 调整搜索参数（时间限制、迭代次数等）")
    print("  3. 尝试其他搜索算法（MCTS、PDVN等）")
    print("  4. 使用可视化工具查看合成树")
    print("\n查看完整教程:")
    print("  - Notebook: docs/tutorials/多步逆合成完整教程.ipynb")
    print("  - 文档: docs/多步逆合成使用指南.md")
    print("  - README: 多步逆合成教程-README.md")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
    except Exception as e:
        print(f"\n\n错误: {e}")
        print("请检查:")
        print("  1. 是否正确安装了 syntheseus")
        print("  2. 是否有网络连接（首次运行需要下载模型）")
        print("  3. 是否有足够的磁盘空间")
        raise
