from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import IPythonConsole

# SMILES字符串
smiles = ""

# 创建分子对象
mol = Chem.MolFromSmiles(smiles)

# 生成2D坐标
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
AllChem.MMFFOptimizeMolecule(mol)

# 保存为图片
Draw.MolToFile(mol, "molecule.png", size=(800, 800)) 