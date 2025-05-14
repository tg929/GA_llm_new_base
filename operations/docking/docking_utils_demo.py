import os
import sys
import argparse
import logging
from tqdm import tqdm
from docking_utils import DockingVina

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "docking_utils_demo.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='多受体qvina2分子对接脚本')
    parser.add_argument('-i', '--input', required=True, help='输入SMILES文件')
    parser.add_argument('-o', '--output_dir', default="output/docking_utils_demo", help='输出目录')
    parser.add_argument('--targets', nargs='+', default=['fa7', 'parp1', '5ht1b', 'jak2', 'braf'], help='受体蛋白列表')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    logger = logging.getLogger("docking_utils_demo")

    # 读取SMILES
    with open(args.input) as f:
        smiles_list = [line.strip().split()[0] for line in f if line.strip()]
    logger.info(f"读取到 {len(smiles_list)} 个分子")

    # 对每个受体蛋白分别对接
    for target in args.targets:
        logger.info(f"开始对接受体: {target}")
        docking = DockingVina(target)
        try:
            affinity_list = docking.predict(smiles_list)
        except Exception as e:
            logger.error(f"对接 {target} 失败: {e}")
            continue
        # 输出结果
        output_file = os.path.join(output_dir, f"docked_{target}.smi")
        with open(output_file, 'w') as fout:
            for smi, score in zip(smiles_list, affinity_list):
                fout.write(f"{smi}\t{score:.4f}\n")
        logger.info(f"受体 {target} 对接完成，结果保存至: {output_file}")

    logger.info("所有受体对接完成！")

if __name__ == "__main__":
    main()
