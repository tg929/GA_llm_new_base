from rdkit import Chem
from rdkit.Chem import Draw, AllChem, QED
import sys
sys.path.append('.')  
# SMILES字符串
#smiles = "C[C@H]1CC(O)(c2ccc(CCNC(=O)N3NNNC3[C@]3(Cn4cc(-c5ccc6ccc7cccc8ccc5c6c78)nn4)CC3(C)C)c(F)c2)CN(C)[C@H]1C"
smiles = "[H]c1c([H])c(C([H])([H])C([H])([H])N([H])C(=O)n2n([H])n([H])n([H])c2([H])C2(C([H])([H])n3nnc(-c4c([H])c([H])c5c([H])c([H])c6c([H])c([H])c([H])c7c([H])c([H])c4c5c67)c3[H])C([H])([H])C2(C([H])([H])[H])C([H])([H])[H])c(F)c([H])c1Br"
# 创建分子对象
mol = Chem.MolFromSmiles(smiles)

# 生成2D坐标
AllChem.Compute2DCoords(mol)

# 计算QED
qed_score = QED.qed(mol)


print(f"QED: {qed_score:.3f}")


# 绘制并保存为图片
Draw.MolToFile(
    mol, 
    "molecule_15.2.png", 
    size=(600, 600), 
    kekulize=True, 
    wedgeBonds=True, 
    legend=f"QED: {qed_score:.3f}"
)