import argparse
import os
import numpy as np
import sys
import time
import logging
import subprocess
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from operations.docking.docking_utils import DockingVina
from rdkit import Chem
import pandas as pd
from operations.scoring.eval import calculate_composite_score
from fragment_GPT.utils.chem_utils import get_qed, get_sa
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
def setup_logging(output_dir, generation_num):
    os.makedirs(output_dir, exist_ok=True)  
    log_file = os.path.join(output_dir, f"ga_evolution_{generation_num}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("GA_llm_new")

def run_decompose(input_file, output_prefix, logger):    
    logger.info(f"开始分子分解: {input_file}")    
    decompose_dir = os.path.join(PROJECT_ROOT, "datasets/decompose/decompose_results")
    os.makedirs(decompose_dir, exist_ok=True)    
   
    output_file = os.path.join(decompose_dir, f"frags_result_{output_prefix}.smi")
    output_file2 = os.path.join(decompose_dir, f"frags_seq_{output_prefix}.smi")
    output_file3 = os.path.join(decompose_dir, f"truncated_frags_{output_prefix}.smi")
    output_file4 = os.path.join(decompose_dir, f"decomposable_mols_{output_prefix}.smi")    
   
    decompose_script = os.path.join(PROJECT_ROOT, "datasets/decompose/demo_frags.py")
    cmd = [
        "python", decompose_script,
        "-i", input_file,
        "-o", output_file,
        "-o2", output_file2,
        "-o3", output_file3,
        "-o4", output_file4
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子分解失败: {process.stderr}")
        raise Exception("分子分解失败")
    
    logger.info(f"分子分解完成，生成文件: {output_file3}")
    return output_file3

def run_gpt_generation(input_file, output_prefix, gen_num, logger):    
    logger.info(f"开始GPT生成: {input_file}")    
    
    output_dir = os.path.join(PROJECT_ROOT, "fragment_GPT/output")
    os.makedirs(output_dir, exist_ok=True)    
   
    output_file = os.path.join(output_dir, f"crossovered{gen_num}_frags_new_{gen_num}.smi")    
    
    generate_script = os.path.join(PROJECT_ROOT, "fragment_GPT/generate_all.py")
    cmd = [
        "python", generate_script,
        "--input_file", input_file,
        "--device", "0",
        "--seed", str(gen_num)
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"GPT生成失败: {process.stderr}")
        raise Exception("GPT生成失败")
    
    if not os.path.exists(output_file):
        logger.warning(f"警告: 预期的输出文件 {output_file} 不存在，尝试查找替代文件...")
        alternative_file = os.path.join(output_dir, f"crossovered{output_prefix}_frags_new_{gen_num}.smi")
        if os.path.exists(alternative_file):
            logger.info(f"找到替代文件: {alternative_file}")
            output_file = alternative_file
        else:
            dir_files = [f for f in os.listdir(output_dir) if f.endswith(f"_new_{gen_num}.smi")]
            if dir_files:
                newest_file = max(dir_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
                output_file = os.path.join(output_dir, newest_file)
                logger.info(f"找到最新生成的文件: {output_file}")
            else:
                raise Exception(f"找不到GPT生成的输出文件,生成可能失败")
    
    logger.info(f"GPT生成完成,输出文件: {output_file}")
    return output_file

def run_crossover(source_file, llm_file, output_file, gen_num, num_crossovers, logger):    
    logger.info(f"开始分子交叉: 源文件 {source_file}, LLM生成文件 {llm_file}, 交叉生成新个体数目 {num_crossovers}")    
   
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)    
    
    crossover_script = os.path.join(PROJECT_ROOT, "operations/crossover/crossover_demo_finetune.py")
    cmd = [
        "python", crossover_script,
        "--source_compound_file", source_file,
        "--llm_generation_file", llm_file,
        "--output_file", output_file,
        "--crossover_attempts", str(num_crossovers),
        "--output_dir", os.path.dirname(output_file)
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子交叉失败: {process.stderr}")
        raise Exception("分子交叉失败")
    
    logger.info(f"分子交叉完成，生成文件: {output_file}")
    return output_file

def run_mutation(input_file, llm_file, output_file, num_mutations, logger):   
    logger.info(f"开始分子变异: 输入文件 {input_file}, LLM生成文件 {llm_file}, 变异生成新个体数目 {num_mutations}")    
    
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)    
    
    mutation_script = os.path.join(PROJECT_ROOT, "operations/mutation/mutation_demo_finetune.py")
    cmd = [
        "python", mutation_script,
        "--input_file", input_file,
        "--llm_generation_file", llm_file,
        "--output_file", output_file,
        "--num_mutations", str(num_mutations),
        "--output_dir", os.path.dirname(output_file)
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子变异失败: {process.stderr}")
        raise Exception("分子变异失败")
    
    logger.info(f"分子变异完成，生成文件: {output_file}")
    return output_file

def run_filter(input_file, output_file, logger, args):   
    logger.info(f"开始分子过滤: {input_file}")    
    
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)    
   
    filter_params = []    
   
    if args.LipinskiStrictFilter:
        filter_params.extend(["--LipinskiStrictFilter"])
    if args.LipinskiLenientFilter:
        filter_params.extend(["--LipinskiLenientFilter"])
    if args.GhoseFilter:
        filter_params.extend(["--GhoseFilter"])
    if args.GhoseModifiedFilter:
        filter_params.extend(["--GhoseModifiedFilter"])
    if args.MozziconacciFilter:
        filter_params.extend(["--MozziconacciFilter"])
    if args.VandeWaterbeemdFilter:
        filter_params.extend(["--VandeWaterbeemdFilter"])
    if args.PAINSFilter:
        filter_params.extend(["--PAINSFilter"])
    if args.NIHFilter:
        filter_params.extend(["--NIHFilter"])
    if args.BRENKFilter:
        filter_params.extend(["--BRENKFilter"])
    if args.No_Filters:
        filter_params.extend(["--No_Filters"])      
    if args.alternative_filter:
        for filter_entry in args.alternative_filter:
            filter_params.extend(["--alternative_filter", filter_entry])  
    
    if not filter_params and not args.No_Filters:
        logger.warning("没有指定任何过滤器参数，将使用默认过滤器")   
   
    filter_script = os.path.join(PROJECT_ROOT, "operations/filter/filter_demo.py")
    cmd = [
        "python", filter_script,
        "--input", input_file,
        "--output", output_file
    ]  
    
    cmd.extend(filter_params)
    
    logger.info(f"执行过滤命令: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子过滤失败: {process.stderr}")
        raise Exception("分子过滤失败")
    
    logger.info(f"分子过滤完成，生成文件: {output_file}")
    return output_file

def dock_molecule(mol_idx, mol_smiles, args, temp_dir, logger):   
    try:
        # 临时文件
        temp_input = os.path.join(temp_dir, f"temp_input_{mol_idx}.smi")
        temp_output = os.path.join(temp_dir, f"temp_output_{mol_idx}.smi")        
       
        with open(temp_input, 'w') as f:
            f.write(mol_smiles)        
       
        docking_script = os.path.join(PROJECT_ROOT, "operations/docking/docking_demo.py")
        cmd = [
            "python", docking_script,
            "-i", temp_input,
            "-r", args.receptor_file,
            "-o", temp_output,
            "-m", args.mgltools_path,
            "--max_failures", "5"
        ]       
       
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.warning(f"分子 {mol_idx} 对接失败: {process.stderr}")
            return None       
       
        if os.path.exists(temp_output):
            with open(temp_output, 'r') as f:
                result = f.read().strip()
            if result:
                return result
        
        return None
        
    except Exception as e:
        logger.error(f"分子 {mol_idx} 对接过程出错: {str(e)}")
        return None
    finally:        
        try:
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except:
            pass

def run_docking(input_file, output_file, receptor_file, mgltools_path, logger, num_processors=1, multithread_mode="serial"):
    """运行分子对接，支持并行处理"""
    logger.info(f"开始分子对接: {input_file}, 处理器数量: {num_processors}, 模式: {multithread_mode}")  
    
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)   
   
    available_cpus = multiprocessing.cpu_count()
    if num_processors == -1 or num_processors > available_cpus:
        num_processors = available_cpus
        logger.info(f"自动设置使用所有可用的CPU核心: {num_processors}")    
    
    if num_processors > 1 and multithread_mode == "serial":
        logger.info(f"检测到使用多核({num_processors})但模式为serial,自动切换为multithreading模式")
        multithread_mode = "multithreading"       
   
    if multithread_mode == "serial" or num_processors == 1:
        logger.info("使用串行模式进行对接")
        docking_script = os.path.join(PROJECT_ROOT, "operations/docking/docking_demo.py")
        cmd = [
            "python", docking_script,
            "-i", input_file,
            "-r", receptor_file,
            "-o", output_file,
            "-m", mgltools_path,
            "--max_failures", "5"
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"分子对接失败: {process.stderr}")
            raise Exception("分子对接失败")
        
        logger.info(f"分子对接完成，生成文件: {output_file}")
        return output_file    
    
    logger.info(f"使用并行模式进行对接，处理器数量: {num_processors}")   
   
    with open(input_file, 'r') as f:
        molecules = [line for line in f.readlines() if line.strip()]
    
    total_molecules = len(molecules)
    logger.info(f"共有 {total_molecules} 个分子需要对接")
    
    # 临时目录存放分割后的文件
    temp_dir = os.path.join(output_dir, "temp_docking")
    os.makedirs(temp_dir, exist_ok=True)
    
   
    dock_func = partial(dock_molecule, args=argparse.Namespace(
        receptor_file=receptor_file,
        mgltools_path=mgltools_path
    ), temp_dir=temp_dir, logger=logger)
    
    # 计算每个处理器应该处理的分子数量，确保负载平衡
    molecules_per_processor = max(1, total_molecules // num_processors)
    
    # 并行执行对接
    results = []
    start_time = time.time()   
    
    batch_size = max(1, min(100, molecules_per_processor))    
   
    molecule_batches = []
    for i in range(0, total_molecules, batch_size):
        end = min(i + batch_size, total_molecules)
        molecule_batches.append((i, molecules[i:end]))
    
    logger.info(f"将 {total_molecules} 个分子分为 {len(molecule_batches)} 批进行处理，每批大约 {batch_size} 个分子")
    
    # 优化：使用批处理方式进行对接
    if multithread_mode == "multithreading":
        logger.info(f"使用多线程模式，线程数: {num_processors}")
        with ThreadPoolExecutor(max_workers=num_processors) as executor:
           
            future_to_idx = {}
            for batch_idx, (start_idx, batch) in enumerate(molecule_batches):
                for mol_idx, mol in enumerate(batch):
                    future = executor.submit(dock_func, start_idx + mol_idx, mol)
                    future_to_idx[future] = start_idx + mol_idx           
           
            completed = 0
            successful = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                completed += 1
                if result:
                    results.append(result)
                    successful += 1               
               
                if completed % max(1, total_molecules // 20) == 0 or completed == total_molecules:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / completed) * (total_molecules - completed) if completed > 0 else 0
                    logger.info(f"已完成: {completed}/{total_molecules} ({completed/total_molecules*100:.1f}%), "
                               f"成功: {successful}/{completed} ({successful/completed*100:.1f}% 成功率), "
                               f"耗时: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒")
    else:  
        logger.info(f"使用多进程模式，进程数: {num_processors}")        
        mp_context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=num_processors, mp_context=mp_context) as executor:
           
            future_to_idx = {}
            for batch_idx, (start_idx, batch) in enumerate(molecule_batches):
                for mol_idx, mol in enumerate(batch):
                    future = executor.submit(dock_func, start_idx + mol_idx, mol)
                    future_to_idx[future] = start_idx + mol_idx        
          
            completed = 0
            successful = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                completed += 1
                if result:
                    results.append(result)
                    successful += 1               
               
                if completed % max(1, total_molecules // 20) == 0 or completed == total_molecules:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / completed) * (total_molecules - completed) if completed > 0 else 0
                    logger.info(f"已完成: {completed}/{total_molecules} ({completed/total_molecules*100:.1f}%), "
                               f"成功: {successful}/{completed} ({successful/completed*100:.1f}% 成功率), "
                               f"耗时: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒")
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"对接计算完成，总耗时: {total_time:.2f}秒，"
               f"平均每个分子: {total_time/total_molecules:.2f}秒，"
               f"总成功率: {len(results)/total_molecules*100:.1f}%")   
    
    with open(output_file, 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    logger.info(f"并行对接完成，成功对接 {len(results)}/{total_molecules} 个分子，结果保存至: {output_file}")
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return output_file

def run_analysis(input_file, output_prefix, gen_num, logger):   
    logger.info(f"开始对接结果分析: {input_file}")    
   
    output_dir = os.path.dirname(input_file)   
    
    analysis_script = os.path.join(PROJECT_ROOT, "operations/docking/analyse_result_0.py")
    cmd = [
        "python", analysis_script,
        "--input", input_file,
        "--output", output_dir,
        "--prefix", f"generation_{gen_num}"
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"对接结果分析失败: {process.stderr}")
        raise Exception("对接结果分析失败")
    
    logger.info(f"对接结果分析完成，结果保存至: {output_dir}/generation_{gen_num}_stats.txt")
    return f"{output_dir}/generation_{gen_num}_sorted.smi"

def load_and_score_population(docked_file):
    """
    读取docked.smi,计算QED、SA,并用新评分函数打分。
    返回DataFrame,包含smiles, DS, QED, SA, composite_score。
    """
    smiles_list = []
    ds_list = []
    with open(docked_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                smiles_list.append(parts[0])
                ds_list.append(float(parts[1]))
    qed_list = []
    sa_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            qed = get_qed(mol)
            sa = -(9*get_sa(mol)-10)
        else:
            qed = 0.0
            sa = 0.0
        qed_list.append(qed)
        sa_list.append(sa)
    df = pd.DataFrame({
        'smiles': smiles_list,
        'DS': ds_list,
        'QED': qed_list,
        'SA': sa_list
    })
    df = calculate_composite_score(df)
    return df

def calculate_and_print_stats(docking_output, generation_num, logger):    
    try:
        df = load_and_score_population(docking_output)
    except Exception as e:
        logger.error(f"读取对接结果文件失败: {str(e)}")
        return
    if df.empty:
        logger.warning("对接结果中没有发现有效分数")
        return       
    # 只输出一份原始DS统计
    ds_sorted = df.sort_values(by='DS', ascending=True)  # DS越小越好
    ds_mean = ds_sorted['DS'].mean()
    ds_top1 = ds_sorted['DS'].iloc[0] if len(ds_sorted) >= 1 else None
    ds_top10_mean = ds_sorted['DS'].iloc[:10].mean() if len(ds_sorted) >= 10 else ds_sorted['DS'].mean()
    ds_top20_mean = ds_sorted['DS'].iloc[:20].mean() if len(ds_sorted) >= 20 else ds_sorted['DS'].mean()
    ds_top50_mean = ds_sorted['DS'].iloc[:50].mean() if len(ds_sorted) >= 50 else ds_sorted['DS'].mean()
    ds_top100_mean = ds_sorted['DS'].iloc[:100].mean() if len(ds_sorted) >= 100 else ds_sorted['DS'].mean()
    ds_message = (
        f"\n==================== Generation {generation_num} QVina对接分数统计 ====================\n"
        f"总分子数: {len(ds_sorted)}\n"
        f"所有分子DS均值: {ds_mean:.4f}\n"
        f"Top1 DS: {ds_top1:.4f}\n"
        f"Top10 DS均值: {ds_top10_mean:.4f}\n"
        f"Top20 DS均值: {ds_top20_mean:.4f}\n"
        f"Top50 DS均值: {ds_top50_mean:.4f}\n"
        f"Top100 DS均值: {ds_top100_mean:.4f}\n"
        f"========================================================================\n"
    )
    logger.info(ds_message)
    print(ds_message)
    # 生成QED SA composite_score文件，按composite_score降序排序
    qed_sa_file = os.path.join(os.path.dirname(docking_output), f"generation_{generation_num}_qed_sa.smi")
    df_sorted = df.sort_values(by='composite_score', ascending=False)
    with open(qed_sa_file, 'w') as f:
        for i, row in df_sorted.iterrows():
            f.write(f"{row['smiles']}\t{row['QED']:.4f}\t{row['SA']:.4f}\t{row['composite_score']:.4f}\n")
    logger.info(f"QED/SA/综合评分文件已保存: {qed_sa_file}")

def select_seeds_for_next_generation(docking_output, seed_output, top_mols, diversity_mols, logger, elitism_mols=1, prev_elite_mols=None):
    """基于综合评分选择种子分子，支持精英保留机制"""
    logger.info(f"开始选择种子分子(综合评分): 从 {docking_output} 选择 {top_mols} 个适应度种子和 {diversity_mols} 个多样性种子，保留 {elitism_mols} 个精英分子")
    try:
        df = load_and_score_population(docking_output)
    except Exception as e:
        logger.error(f"读取对接结果文件失败: {str(e)}")
        return None
    if df.empty:
        logger.warning("对接结果中没有发现有效分数")
        return None    
    df_sorted = df.sort_values(by='composite_score', ascending=False)
    current_best_mol = df_sorted.iloc[0]['smiles']
    current_best_score = df_sorted.iloc[0]['composite_score']
    if prev_elite_mols:
        prev_best_mol = list(prev_elite_mols.keys())[0]
        prev_best_score = list(prev_elite_mols.values())[0]        
        if current_best_score > prev_best_score:
            new_elite_mols = {current_best_mol: current_best_score}
            logger.info(f"发现更好的分子，更新精英分子:")
            logger.info(f"上一代精英分子: {prev_best_mol} (得分: {prev_best_score})")
            logger.info(f"新的精英分子: {current_best_mol} (得分: {current_best_score})")
        else:
            new_elite_mols = {prev_best_mol: prev_best_score}
            logger.info(f"保留上一代精英分子:")
            logger.info(f"当前代最好分子: {current_best_mol} (得分: {current_best_score})")
            logger.info(f"保留的精英分子: {prev_best_mol} (得分: {prev_best_score})")
    else:
        new_elite_mols = {current_best_mol: current_best_score}
        logger.info(f"第一代精英分子: {current_best_mol} (得分: {current_best_score})")
    # 从剩余分子中选择适应度种子（排除已选择的精英分子）
    remaining_df = df_sorted[~df_sorted['smiles'].isin(new_elite_mols.keys())]
    fitness_seeds = remaining_df.iloc[:top_mols]['smiles'].tolist()
    logger.info(f"已选择 {len(fitness_seeds)} 个适应度种子")
    # 选择多样性种子（此处仍可用原有字符串距离算法）
    diversity_seeds = []
    remaining_smiles = remaining_df.iloc[top_mols:]['smiles'].tolist()
    if diversity_mols > 0 and remaining_smiles:
        import random
        selected_indices = []
        first_idx = np.random.randint(0, len(remaining_smiles))
        selected_indices.append(first_idx)
        diversity_seeds.append(remaining_smiles[first_idx])
        for _ in range(min(diversity_mols - 1, len(remaining_smiles) - 1)):
            max_min_dist = -1
            best_idx = -1
            for i in range(len(remaining_smiles)):
                if i in selected_indices:
                    continue
                min_dist = float('inf')
                for j in selected_indices:
                    dist = sum(a != b for a, b in zip(remaining_smiles[i], remaining_smiles[j]))
                    min_dist = min(min_dist, dist)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            if best_idx != -1:
                selected_indices.append(best_idx)
                diversity_seeds.append(remaining_smiles[best_idx])
    logger.info(f"已选择 {len(diversity_seeds)} 个多样性种子")
    all_seeds = list(new_elite_mols.keys()) + fitness_seeds + diversity_seeds
    with open(seed_output, 'w') as f:
        for mol in all_seeds:
            f.write(f"{mol}\n")
    logger.info(f"种子选择完成，共选择 {len(all_seeds)} 个分子，保存至: {seed_output}")
    return seed_output, new_elite_mols

def run_docking_qvina(input_file, output_dir, targets, logger):
    """调用docking_utils_demo.py进行多受体qvina02对接"""
    logger.info(f"开始qvina02多受体对接: {input_file} -> {output_dir} (受体: {targets})")
    
    # 添加SMILES格式验证和过滤
    filtered_file = os.path.join(output_dir, "filtered_input.smi")
    valid_molecules = []
    
    with open(input_file, 'r') as f:
        for line in f:
            smiles = line.strip()
            if not smiles:
                continue
                
            # 检查是否包含*号
            if '*' in smiles:
                logger.warning(f"跳过包含*号的SMILES: {smiles}")
                continue
                
            # 尝试转换为RDKit分子对象
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_molecules.append(smiles)
                else:
                    logger.warning(f"无效的SMILES字符串: {smiles}")
            except Exception as e:
                logger.warning(f"处理SMILES时出错: {smiles}, 错误: {str(e)}")
    
    # 保存过滤后的分子
    with open(filtered_file, 'w') as f:
        for smiles in valid_molecules:
            f.write(f"{smiles}\n")
    
    logger.info(f"SMILES过滤完成: 原始分子数 {sum(1 for _ in open(input_file))}, 有效分子数 {len(valid_molecules)}")
    
    # 使用过滤后的文件进行对接
    script_path = os.path.join(PROJECT_ROOT, "operations/docking/docking_utils_demo.py")
    cmd = [
        "python", script_path,
        "-i", filtered_file,
        "-o", output_dir,
        "--targets"
    ] + targets
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        logger.error(f"qvina02多受体对接失败: {process.stderr}")
        raise Exception("qvina02多受体对接失败")
    
    logger.info(f"qvina02多受体对接完成, 结果目录: {output_dir}")
    
    # 清理临时文件
    if os.path.exists(filtered_file):
        os.remove(filtered_file)
    
    return [os.path.join(output_dir, f"docked_{t}.smi") for t in targets]

def run_evolution(generation_num, args, logger, prev_elite_mols=None):
    """执行一次完整的进化迭代，支持精英保留机制"""
    logger.info(f"开始第 {generation_num} 代进化")
    output_base = os.path.join(args.output_dir, f"generation_{generation_num}")
    os.makedirs(output_base, exist_ok=True)

    # 0. 确定当前代的种群文件
    if generation_num == 0:
        current_population = args.initial_population
        # 初代直接docking+scoring
        docking_output = os.path.join(output_base, f"generation_{generation_num}_docked.smi")
        run_docking(current_population, docking_output, args.receptor_file, args.mgltools_path, logger, args.number_of_processors, args.multithread_mode)
        calculate_and_print_stats(docking_output, generation_num, logger)
        # 选seed
        diversity_mols = max(0, args.diversity_mols_to_seed_first_generation - (generation_num * args.diversity_seed_depreciation_per_gen))
        seed_output = os.path.join(output_base, f"generation_{generation_num}_seeds.smi")
        seed_output, new_elite_mols = select_seeds_for_next_generation(
            docking_output, seed_output, args.top_mols_to_seed_next_generation, 
            diversity_mols, logger, args.elitism_mols_to_next_generation
        )
        return seed_output, new_elite_mols
    else:
        # 1. 读取上一代seed，但排除精英分子
        prev_seed_file = os.path.join(args.output_dir, f"generation_{generation_num-1}", f"generation_{generation_num-1}_seeds.smi")
        non_elite_molecules = []
        with open(prev_seed_file, 'r') as f:
            for line in f:
                mol = line.strip()
                if mol and (prev_elite_mols is None or mol not in prev_elite_mols):
                    non_elite_molecules.append(mol)
        
        # 2. 只对非精英分子进行decompose+gpt生成
        temp_seed_file = os.path.join(output_base, "temp_non_elite_seeds.smi")
        with open(temp_seed_file, 'w') as f:
            for mol in non_elite_molecules:
                f.write(f"{mol}\n")
        
        decompose_output = run_decompose(temp_seed_file, f"gen{generation_num}_seed", logger)
        gpt_output = run_gpt_generation(decompose_output, f"gen{generation_num}_seed", generation_num, logger)
        
        # 3. 交叉（只使用非精英分子）
        crossover_output = os.path.join(output_base, f"generation_{generation_num}_crossover.smi")
        run_crossover(temp_seed_file, gpt_output, crossover_output, generation_num, args.num_crossovers, logger)
        
        # 4. 变异（只使用非精英分子）
        mutation_output = os.path.join(output_base, f"generation_{generation_num}_mutation.smi")
        run_mutation(temp_seed_file, gpt_output, mutation_output, args.num_mutations, logger)
        
        # 5. 合并新种群（精英分子 + 新生成的分子）
        new_population_file = os.path.join(output_base, f"generation_{generation_num}_new_population.smi")
        with open(new_population_file, 'w') as fout:
            # 首先写入精英分子（如果有的话）
            if prev_elite_mols:
                for mol, score in prev_elite_mols.items():
                    fout.write(f"{mol}\n")
                logger.info(f"已将上一代精英分子 {list(prev_elite_mols.keys())[0]} (得分: {list(prev_elite_mols.values())[0]}) 加入新种群")
            
            # 然后写入交叉和变异产生的新分子
            for fname in [crossover_output, mutation_output]:
                with open(fname, 'r') as fin:
                    for line in fin:
                        if line.strip():
                            fout.write(line)
        
        # 6. docking+scoring
        docking_output = os.path.join(output_base, f"generation_{generation_num}_docked.smi")
        run_docking(new_population_file, docking_output, args.receptor_file, args.mgltools_path, logger, args.number_of_processors, args.multithread_mode)

        # === 新增：确保精英分子直接保留到docked.smi中 ===
        if prev_elite_mols:
            elite_smiles, elite_score = list(prev_elite_mols.items())[0]
            # 检查精英分子是否已在docked.smi中
            found = False
            with open(docking_output, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0] == elite_smiles:
                        found = True
                        break
            if not found:
                # 直接追加精英分子及其分数
                with open(docking_output, 'a') as f:
                    f.write(f"{elite_smiles}\t{elite_score:.4f}\n")
                logger.info(f"精英分子未在docked.smi中,已直接追加: {elite_smiles} (得分: {elite_score})")

        calculate_and_print_stats(docking_output, generation_num, logger)
        
        # 7. 选seed（精英分子只在新一代top1更优时才更新）
        diversity_mols = max(0, args.diversity_mols_to_seed_first_generation - (generation_num * args.diversity_seed_depreciation_per_gen))
        seed_output = os.path.join(output_base, f"generation_{generation_num}_seeds.smi")

        # 读取docked.smi，比较top1和精英分子分数
        molecules, scores = [], []
        with open(docking_output, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    molecules.append(parts[0])
                    scores.append(float(parts[1]))
        if scores:
            sorted_indices = np.argsort(scores)
            sorted_molecules = [molecules[i] for i in sorted_indices]
            sorted_scores = [scores[i] for i in sorted_indices]
            current_best_mol = sorted_molecules[0]
            current_best_score = sorted_scores[0]
        else:
            current_best_mol = None
            current_best_score = None

        # 比较精英分子和新top1，选择更优的传递
        if prev_elite_mols:
            elite_smiles, elite_score = list(prev_elite_mols.items())[0]
            if current_best_score is not None and current_best_score < elite_score:
                new_elite_mols = {current_best_mol: current_best_score}
                logger.info(f"新一代top1更优,更新精英分子: {current_best_mol} (得分: {current_best_score})")
            else:
                new_elite_mols = {elite_smiles: elite_score}
                logger.info(f"继续保留上一代精英分子: {elite_smiles} (得分: {elite_score})")
        else:
            if current_best_mol is not None:
                new_elite_mols = {current_best_mol: current_best_score}
                logger.info(f"第一代精英分子: {current_best_mol} (得分: {current_best_score})")
            else:
                new_elite_mols = None

        # 选seed时传递new_elite_mols
        seed_output, new_elite_mols = select_seeds_for_next_generation(
            docking_output, seed_output, args.top_mols_to_seed_next_generation, 
            diversity_mols, logger, args.elitism_mols_to_next_generation, new_elite_mols
        )
        
        # 清理临时文件
        if os.path.exists(temp_seed_file):
            os.remove(temp_seed_file)
            
        return seed_output, new_elite_mols

def main():
    
    parser = argparse.ArgumentParser(description='GA_llm - 多受体分子进化与生成流程')    
   
    parser.add_argument('--generations', type=int, default=5, 
                        help='进化代数(不包括第0代,总共会生成6代:generation_0到generation_5)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, 'output'),
                        help='输出目录')
    parser.add_argument('--initial_population', type=str, 
                        default=os.path.join(PROJECT_ROOT, 'datasets/source_compounds/naphthalene_smiles.smi'))   
    
    parser.add_argument('--receptor_file', type=str,
                        default=os.path.join(PROJECT_ROOT, 'tutorial/PARP/4r6eA_PARP1_prepared.pdb'))
    parser.add_argument('--mgltools_path', type=str,
                        default=os.path.join(PROJECT_ROOT, 'mgltools_x86_64Linux2_1.5.6'))
    
    # 进化参数
    parser.add_argument('--num_crossovers', type=int, default=50)
    parser.add_argument('--num_mutations', type=int, default=50)
    parser.add_argument('--number_of_crossovers_first_generation', type=int,
                       help='第0代中通过交叉产生的配体数量,如果未指定则默认使用num_crossovers的值')
    parser.add_argument('--number_of_mutants_first_generation', type=int,
                       help='第0代中通过变异产生的配体数量,如果未指定则默认使用num_mutations的值')
    parser.add_argument('--max_population', type=int, default=0,
                       help='控制每代种群的最大数量,设置为0表示不限制(可能导致种群规模迅速增长）')
    
    # 种子选择参数
    parser.add_argument('--top_mols_to_seed_next_generation', type=int, default=10,
                       help='每代基于适应度选择进入下一代的分子数量')
    parser.add_argument('--diversity_mols_to_seed_first_generation', type=int, default=10,
                       help='第0代基于多样性选择进入下一代的分子数量')
    parser.add_argument('--diversity_seed_depreciation_per_gen', type=int, default=2,
                       help='每代多样性种子数量的递减值')
    parser.add_argument('--elitism_mols_to_next_generation', type=int, default=1,
                       help='每代保留的精英分子数量，这些分子将直接进入下一代而不进行进化操作')
    
    # 并行处理参数
    parser.add_argument('--number_of_processors', '-p', type=int, default=-1,
                        help='用于并行计算的处理器数量。设置为-1表示自动检测并使用所有可用CPU核心(推荐）。')
    parser.add_argument('--multithread_mode', default="multithreading",
                        choices=["mpi", "multithreading", "serial"],
                        help='多线程模式选择: mpi, multithreading, 或 serial。serial模式将忽略处理器数量设置,强制使用单处理器。')
    
    # 过滤器参数
    parser.add_argument('--LipinskiStrictFilter', action='store_true', default=False,
                        help='严格版Lipinski五规则过滤器,筛选口服可用药物。评估分子量、logP、氢键供体和受体数量。要求必须通过所有条件。')
    parser.add_argument('--LipinskiLenientFilter', action='store_true', default=False,
                        help='宽松版Lipinski五规则过滤,筛选口服可用药物。评估分子量、logP、氢键供体和受体数量。允许一个条件不满足。')
    parser.add_argument('--GhoseFilter', action='store_true', default=False,
                        help='Ghose药物相似性过滤器,通过分子量、logP和原子数量进行筛选。')
    parser.add_argument('--GhoseModifiedFilter', action='store_true', default=False,
                        help='修改版Ghose过滤器,将分子量上限从480Da放宽到500Da。设计用于与Lipinski过滤器配合使用。')
    parser.add_argument('--MozziconacciFilter', action='store_true', default=False,
                        help='Mozziconacci药物相似性过滤器,评估可旋转键、环、氧原子和卤素原子的数量。')
    parser.add_argument('--VandeWaterbeemdFilter', action='store_true', default=False,
                        help='筛选可能透过血脑屏障的药物，基于分子量和极性表面积(PSA)。')
    parser.add_argument('--PAINSFilter', action='store_true', default=False,
                        help='PAINS过滤器,用于过滤泛测试干扰化合物,使用子结构搜索。')
    parser.add_argument('--NIHFilter', action='store_true', default=False,
                        help='NIH过滤器,过滤含有不良功能团的分子，使用子结构搜索。')
    parser.add_argument('--BRENKFilter', action='store_true', default=False,
                        help='BRENK前导物相似性过滤器,排除常见假阳性分子。')
    parser.add_argument('--No_Filters', action='store_true', default=False,
                        help='设置为True时,不应用任何过滤器。')
    parser.add_argument('--alternative_filter', action='append',
                        help='添加自定义过滤器，需要提供列表格式：[[过滤器1名称, 过滤器1路径], [过滤器2名称, 过滤器2路径]]')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)    
    
    if args.number_of_processors == -1:
        print(f"将使用动态检测的CPU数量,在每次对接时自动设置")
    else:
        available_cpus = multiprocessing.cpu_count()
        if args.number_of_processors > available_cpus:
            print(f"指定的处理器数量({args.number_of_processors})超过系统可用CPU数量({available_cpus})，将使用所有可用CPU")
            args.number_of_processors = available_cpus
        else:
            print(f"将使用指定的{args.number_of_processors}个CPU进行计算")    
    
    if args.number_of_processors != 1 and args.multithread_mode == "serial":
        print(f"检测到可能使用多核但模式为serial,自动切换为multithreading模式")
        args.multithread_mode = "multithreading"    
   
    # if args.max_population > 0:
    #     # 检查初始种群大小
    #     with open(args.initial_population, 'r') as f:
    #         initial_count = sum(1 for line in f if line.strip())
    #     if initial_count > args.max_population:
    #         limited_file = os.path.join(args.output_dir, "limited_initial_population.smi")
    #         args.initial_population = limit_population_size(args.initial_population, args.max_population, limited_file)
    #         print(f"初始种群已从{initial_count}限制为{args.max_population}")
    
    # 执行多代进化
    # # 先运行第0代（进行交叉和变异操作后再对接）
    # logger = setup_logging(args.output_dir, 0)
    # try:
    #     # 确定第0代使用的交叉变异生成新分子数目
    #     if args.number_of_crossovers_first_generation is None:
    #         args.number_of_crossovers_first_generation = args.num_crossovers
    #         logger.info(f"第0代交叉生成新个体数未指定,使用默认值: {args.num_crossovers}")
        
    #     if args.number_of_mutants_first_generation is None:
    #         args.number_of_mutants_first_generation = args.num_mutations
    #         logger.info(f"第0代变异生成新个体数未指定,使用默认值: {args.num_mutations}")
            
    #     logger.info(f"开始第0代(对初始种群进行交叉变异后对接)")
    #     logger.info(f"第0代将通过交叉生成 {args.number_of_crossovers_first_generation} 个新分子和 通过变异生成{args.number_of_mutants_first_generation} 个新分子")
    #     start_time = time.time()
        
    #     final_output, elite_mols = run_evolution(0, args, logger)
        
    #     end_time = time.time()
    #     logger.info(f"第0代完成,耗时: {end_time - start_time:.2f}秒")
    # except Exception as e:
    #     logger.error(f"第0代失败: {str(e)}")
    #     elite_mols = None
    
    # 执行后续5代进化
    targets = ['fa7', 'parp1', '5ht1b', 'jak2', 'braf']
    #targets = ['parp1']

    #targets = ['jak2', 'braf']  # 只保留未完成的两个受体
    for target in targets:
        logger = setup_logging(os.path.join(args.output_dir, target), 0)
        logger.info(f"==== 开始受体 {target} 的进化优化 ====")
        # 受体相关输出目录
        output_base = os.path.join(args.output_dir, target)
        os.makedirs(output_base, exist_ok=True)
        # 初始化精英分子
        elite_mols = None
        for gen in range(args.generations + 1):
            logger = setup_logging(output_base, gen)
            logger.info(f"开始第 {gen} 代进化 (受体: {target})")
            start_time = time.time()
            # 文件名都带target
            if gen == 0:
                current_population = args.initial_population
                docking_output_dir = os.path.join(output_base, f"generation_{gen}_docking")
                os.makedirs(docking_output_dir, exist_ok=True)
                docked_files = run_docking_qvina(current_population, docking_output_dir, [target], logger)
                docking_output = docked_files[0]
                calculate_and_print_stats(docking_output, gen, logger)
                diversity_mols = max(0, args.diversity_mols_to_seed_first_generation - (gen * args.diversity_seed_depreciation_per_gen))
                seed_output = os.path.join(output_base, f"generation_{gen}_seeds_{target}.smi")
                seed_output, elite_mols = select_seeds_for_next_generation(
                    docking_output, seed_output, args.top_mols_to_seed_next_generation, 
                    diversity_mols, logger, args.elitism_mols_to_next_generation
                )
            else:
                prev_seed_file = os.path.join(output_base, f"generation_{gen-1}_seeds_{target}.smi")
                non_elite_molecules = []
                with open(prev_seed_file, 'r') as f:
                    for line in f:
                        mol = line.strip()
                        if mol and (elite_mols is None or mol not in elite_mols):
                            non_elite_molecules.append(mol)
                temp_seed_file = os.path.join(output_base, f"temp_non_elite_seeds_{target}.smi")
                with open(temp_seed_file, 'w') as f:
                    for mol in non_elite_molecules:
                        f.write(f"{mol}\n")
                decompose_output = run_decompose(temp_seed_file, f"gen{gen}_seed_{target}", logger)
                gpt_output = run_gpt_generation(decompose_output, f"gen{gen}_seed_{target}", gen, logger)
                crossover_output = os.path.join(output_base, f"generation_{gen}_crossover_{target}.smi")
                run_crossover(temp_seed_file, gpt_output, crossover_output, gen, args.num_crossovers, logger)
                mutation_output = os.path.join(output_base, f"generation_{gen}_mutation_{target}.smi")
                run_mutation(temp_seed_file, gpt_output, mutation_output, args.num_mutations, logger)
                new_population_file = os.path.join(output_base, f"generation_{gen}_new_population_{target}.smi")
                with open(new_population_file, 'w') as fout:
                    if elite_mols:
                        for mol, score in elite_mols.items():
                            fout.write(f"{mol}\n")
                        logger.info(f"已将上一代精英分子 {list(elite_mols.keys())[0]} (得分: {list(elite_mols.values())[0]}) 加入新种群")
                    for fname in [crossover_output, mutation_output]:
                        with open(fname, 'r') as fin:
                            for line in fin:
                                if line.strip():
                                    fout.write(line)
                docking_output_dir = os.path.join(output_base, f"generation_{gen}_docking")
                os.makedirs(docking_output_dir, exist_ok=True)
                docked_files = run_docking_qvina(new_population_file, docking_output_dir, [target], logger)
                docking_output = docked_files[0]
                # === 精英分子直接保留到docked.smi中 ===
                if elite_mols:
                    elite_smiles, elite_score = list(elite_mols.items())[0]
                    found = False
                    with open(docking_output, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2 and parts[0] == elite_smiles:
                                found = True
                                break
                    if not found:
                        with open(docking_output, 'a') as f:
                            f.write(f"{elite_smiles}\t{elite_score:.4f}\n")
                        logger.info(f"精英分子未在docked.smi中,已直接追加: {elite_smiles} (得分: {elite_score})")
                calculate_and_print_stats(docking_output, gen, logger)
                diversity_mols = max(0, args.diversity_mols_to_seed_first_generation - (gen * args.diversity_seed_depreciation_per_gen))
                seed_output = os.path.join(output_base, f"generation_{gen}_seeds_{target}.smi")
                molecules, scores = [], []
                with open(docking_output, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            molecules.append(parts[0])
                            scores.append(float(parts[1]))
                if scores:
                    sorted_indices = np.argsort(scores)
                    sorted_molecules = [molecules[i] for i in sorted_indices]
                    sorted_scores = [scores[i] for i in sorted_indices]
                    current_best_mol = sorted_molecules[0]
                    current_best_score = sorted_scores[0]
                else:
                    current_best_mol = None
                    current_best_score = None
                if elite_mols:
                    elite_smiles, elite_score = list(elite_mols.items())[0]
                    if current_best_score is not None and current_best_score < elite_score:
                        new_elite_mols = {current_best_mol: current_best_score}
                        logger.info(f"新一代top1更优,更新精英分子: {current_best_mol} (得分: {current_best_score})")
                    else:
                        new_elite_mols = {elite_smiles: elite_score}
                        logger.info(f"继续保留上一代精英分子: {elite_smiles} (得分: {elite_score})")
                else:
                    if current_best_mol is not None:
                        new_elite_mols = {current_best_mol: current_best_score}
                        logger.info(f"第一代精英分子: {current_best_mol} (得分: {current_best_score})")
                    else:
                        new_elite_mols = None
                seed_output, elite_mols = select_seeds_for_next_generation(
                    docking_output, seed_output, args.top_mols_to_seed_next_generation, 
                    diversity_mols, logger, args.elitism_mols_to_next_generation, new_elite_mols
                )
                if os.path.exists(temp_seed_file):
                    os.remove(temp_seed_file)
            end_time = time.time()
            logger.info(f"第 {gen} 代进化完成,耗时: {end_time - start_time:.2f}秒")
            logger.info(f"结果保存至: {output_base}")
        logger.info(f"==== 受体 {target} 的进化优化全部完成 ====")

if __name__ == "__main__":
    main()
