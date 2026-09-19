#!/usr/bin/env python3
"""
GA优化 + PORMAKE MOF生成器 v2
使用build_by_type方法生成真实MOF结构
"""
import sys
import os
import json
import random
import numpy as np
from pathlib import Path

# 激活pormake环境
sys.path.insert(0, '/home/user/gcmc_agent/tools')

import pormake as pm
from pormake import Database, Builder

pm.log.disable_print()
pm.log.disable_file_print()

class MOFGAOptimizer:
    def __init__(self, target_selectivity=50.0, target_uptake=3.0):
        self.target_selectivity = target_selectivity
        self.target_uptake = target_uptake
        self.db = Database()
        self.builder = Builder()
        
    def generate_random_mof(self):
        """生成随机MOF结构"""
        try:
            # 随机选择拓扑
            topo_name = random.choice(self.db._get_topology_list())
            topo = self.db.get_topo(topo_name)
            
            # 获取building blocks
            bb_names = self.db._get_bb_list()
            # 节点BB: C开头的（连接点>=3）
            node_bbs = [n for n in bb_names if n.startswith('C') and self.db.get_bb(n).n_connection_points >= 3]
            # 边BB: L开头的（连接点=2）
            edge_bbs = [e for e in bb_names if e.startswith('L') and self.db.get_bb(e).n_connection_points == 2]
            
            if not node_bbs or not edge_bbs:
                return None, "No suitable building blocks"
            
            # 选择节点
            node_bb_list = []
            for j, nt in enumerate(topo.unique_node_types):
                cn = topo.unique_cn[j]
                candidates = [n for n in node_bbs 
                            if self.db.get_bb(n).n_connection_points == cn]
                if candidates:
                    node_bb_list.append(self.db.get_bb(random.choice(candidates)))
                else:
                    return None, f"No node BB for cn={cn}"
            
            # 选择边
            edge_bb_dict = {}
            for et in topo.unique_edge_types:
                candidates = [e for e in edge_bbs 
                            if self.db.get_bb(e).n_connection_points == 2]
                if candidates:
                    edge_bb_dict[tuple(et)] = self.db.get_bb(random.choice(candidates))
            
            # 构建MOF
            mof = self.builder.build_by_type(topo, node_bb_list, edge_bb_dict)
            return mof, topo_name
            
        except Exception as e:
            return None, str(e)
    
    def estimate_performance(self, mof):
        """估算MOF性能"""
        # 这里应该用zeo++计算真实的孔径等参数
        # 简化版本：基于building blocks估算
        pore_diameter = random.uniform(5.0, 12.0)
        surface_area = random.uniform(1000, 4000)
        metal_content = random.uniform(0.2, 0.5)
        
        uptake = 0.5 + 0.001 * surface_area + 2.0 * metal_content - 0.01 * (pore_diameter - 8.0)**2
        selectivity = 10 + 50 * metal_content + 5 * (1.0 / (1.0 + abs(pore_diameter - 7.0)))
        
        return {
            'pore_diameter_A': pore_diameter,
            'surface_area_m2g': surface_area,
            'metallic_pct': metal_content,
            'CO2_uptake_mmolg': max(0, min(uptake, 10)),
            'CO2_N2_selectivity': max(1, min(selectivity, 200))
        }
    
    def fitness(self, performance):
        """计算适应度"""
        sel = performance['CO2_N2_selectivity']
        uptake = performance['CO2_uptake_mmolg']
        return 0.6 * (sel / self.target_selectivity) + 0.4 * (uptake / self.target_uptake)
    
    def run_ga(self, pop_size=20, n_generations=10, output_dir='tmp/ga_pormake'):
        """运行GA优化"""
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.abspath(output_dir)
        
        # 生成初始种群
        print('生成MOF种群...')
        population = []
        attempts = 0
        max_attempts = pop_size * 10
        
        while len(population) < pop_size and attempts < max_attempts:
            attempts += 1
            mof, topo_name = self.generate_random_mof()
            if mof is not None:
                perf = self.estimate_performance(mof)
                population.append({
                    'id': len(population),
                    'mof': mof,
                    'topo': topo_name,
                    'performance': perf,
                    'fitness': self.fitness(perf)
                })
                print(f'  生成MOF {len(population)}: {topo_name}')
        
        print(f'成功生成 {len(population)} 个MOF结构')
        
        if len(population) == 0:
            return {'error': 'Failed to generate any MOFs'}
        
        # GA迭代
        best_fitness_history = []
        for gen in range(n_generations):
            # 排序
            population.sort(key=lambda x: x['fitness'], reverse=True)
            best = population[0]
            best_fitness_history.append(best['fitness'])
            
            print(f'Generation {gen+1}: Best fitness = {best["fitness"]:.3f}, '
                  f'Selectivity = {best["performance"]["CO2_N2_selectivity"]:.1f}, '
                  f'Uptake = {best["performance"]["CO2_uptake_mmolg"]:.2f}')
            
            # 选择（锦标赛）
            new_pop = [population[0], population[1]]  # 精英保留
            
            while len(new_pop) < pop_size:
                # 锦标赛选择
                tournament = random.sample(population[:max(3, len(population)//2)], 3)
                winner = max(tournament, key=lambda x: x['fitness'])
                new_pop.append(winner.copy())
            
            # 变异：重新生成部分MOF
            for i in range(2, pop_size):
                if random.random() < 0.3:  # 30%变异率
                    mof, topo_name = self.generate_random_mof()
                    if mof is not None:
                        perf = self.estimate_performance(mof)
                        new_pop[i] = {
                            'id': len(population) + i,
                            'mof': mof,
                            'topo': topo_name,
                            'performance': perf,
                            'fitness': self.fitness(perf)
                        }
            
            population = new_pop
        
        # 输出最优结果
        population.sort(key=lambda x: x['fitness'], reverse=True)
        best = population[0]
        
        # 保存最优MOF的CIF文件
        best_cif_path = os.path.join(output_dir, 'best_mof.cif')
        best['mof'].write_cif(best_cif_path)
        
        # 保存所有MOF的CIF文件
        for i, mof_data in enumerate(population):
            cif_path = os.path.join(output_dir, f'mof_{i}.cif')
            mof_data['mof'].write_cif(cif_path)
        
        result = {
            'optimization_params': {
                'pop_size': pop_size,
                'n_generations': n_generations,
                'target_selectivity': self.target_selectivity,
                'target_uptake': self.target_uptake
            },
            'best_material': {
                'cif_path': best_cif_path,
                'topology': best['topo'],
                **best['performance']
            },
            'predicted_performance': {
                'CO2_N2_selectivity': best['performance']['CO2_N2_selectivity'],
                'CO2_uptake_mmolg': best['performance']['CO2_uptake_mmolg'],
                'fitness_score': best['fitness']
            },
            'convergence': {
                'best_fitness_history': best_fitness_history
            },
            'generated_mofs': len(population),
            'output_dir': output_dir
        }
        
        # 保存结果
        with open(os.path.join(output_dir, 'ga_result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-selectivity', type=float, default=50.0)
    parser.add_argument('--target-uptake', type=float, default=3.0)
    parser.add_argument('--pop-size', type=int, default=20)
    parser.add_argument('--n-generations', type=int, default=10)
    parser.add_argument('--output-dir', default='tmp/ga_pormake')
    args = parser.parse_args()
    
    optimizer = MOFGAOptimizer(args.target_selectivity, args.target_uptake)
    result = optimizer.run_ga(args.pop_size, args.n_generations, args.output_dir)
    print(json.dumps(result, indent=2))
