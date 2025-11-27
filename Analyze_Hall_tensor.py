"""
输入操作的空间与自旋变换矩阵，即可得到电导与自旋电导的约束关系
Author: Jianting Dong (董建艇)
Date: 2025-11-25
Version: 2.3
"""

import sympy as sp
import argparse
import re
import Grab_from_spin_space_group as Gs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse the Hall tensors with the input operation 完整输入格式参考："\
        "python Analyse_Hall_tensor_v2.1.py -op=-x,y,x+z,1,-mx+my,mx-my,-mz -p=1"
    )
    parser.add_argument(
        "-f",
        "--input_file",
        type=str,
        required=False,
        help="包含对称操作的文件路径,SOC下文件每行格式为：x,y,z,1,mx,my,mz。非SOC下的文件为find_spin_group上的RAW_data",
        dest="input_file",
    )
    parser.add_argument(
        "-noso",
        "--noSOC",
        action="store_true",
        required=False,
        help="Whether the input file is for non-SOC case",
        default=False,
        dest="noso",
    )
    args = parser.parse_args()
    return args


def get_space_f(operation):
    TERM_PATTERN = re.compile(r'([+-]?)(\w+)')
    VAR_MAP = {'x': 0, 'y': 1, 'z': 2}
    def express(expr):
        return ((match.group(2), -1 if match.group(1)=='-' else 1) 
           for match in TERM_PATTERN.finditer(expr))
    def matrix_row(terms):
        row = [0,0,0]
        for var, coff in terms:
            try:
                idx = VAR_MAP[var]
                row[idx] += coff
            except KeyError:
                raise ValueError(f'{var} 这个输入格式是有误的')
        return row
    opera_list = [c.strip() for c in operation.split(' ')][1].split(',')
    if len(opera_list) != 4:
        raise ValueError("要求4个参数，逗号分隔: x, y, z, +1或者-1")
    R_space = [matrix_row(express(opera_list[i])) for i in range(3)]
    
    f = int(opera_list[3].strip())
    return sp.Matrix(R_space), f


def get_MPG_operators(input_file):
    """
    从磁点群文件中获取自旋和空间变换矩阵
    参数:
        input_file: 磁点群文件路径
    返回:
        R_spins: 自旋变换矩阵列表
        R_spaces: 空间变换矩阵列表
        fs: 时间反演系数列表
    """
    R_spaces_Frac, R_spins_Frac, R_spins, R_spaces, fs = [], [], [], [], []
    with open(input_file, 'r', encoding='utf-8') as f:
        operations = f.readlines()
        f.close()
        # 保留字符串形式以确保精确转换
        cell_data = [[float(i) for i in j.strip().split()] for j in operations[:3]]
        std_Cell = Gs.precise_cell(cell_data)

    for operation in operations[3:]:
        R_space_frac, f = get_space_f(operation.strip())
        R_spaces_Frac.append(R_space_frac)
        R_spins_Frac.append(f * R_space_frac.det() * R_space_frac)

        R_space = Gs.transform_frac_to_cart(R_space_frac, std_Cell)
        R_spaces.append(R_space.applyfunc(Gs.precise_num))
        R_spin = f * R_space.det() * R_space
        R_spins.append(R_spin.applyfunc(Gs.precise_num))
        fs.append(f)

    return R_spins_Frac, R_spaces_Frac, R_spins, R_spaces, fs


def define_symbolic_variables():
    """
    定义符号变量，用于表示电导(CC)和自旋电导(SC)  
    返回:
        sigmaC_odd: 时间反演奇对称HC的符号变量
        sigmaC_even: 时间反演偶对称HC的符号变量
        sigmaS_odd: 时间反演奇对称SHC的符号变量
        sigmaS_even: 时间反演偶对称SHC的符号变量
    """
    # HC: 3x3 电导张量，索引顺序：σ_{current}{field}
    sigmaC_odd = {} # 时间反演奇
    sigmaC_even = {}  # 时间反演偶
    for i in ['x','y','z']:  # i: 电流方向
        for j in ['x','y','z']:  # j: 电场方向
            sigmaC_odd[f"{i}{j}"] = sp.symbols(f"σC_odd_{i}{j}")
            sigmaC_even[f"{i}{j}"] = sp.symbols(f"σC_even_{i}{j}")

    # SHC: 3x3x3 自旋电导张量，索引顺序：σ^{spin}_{current}{field}
    sigmaS_odd = {} # 时间反演奇
    sigmaS_even = {}  # 时间反演偶
    for k in ['X','Y','Z']:  # k: 自旋极化方向
        for i in ['x','y','z']:  # i: 电流方向
            for j in ['x','y','z']:  # j: 电场方向
                # 使用原始定义的符号，确保一致性
                sigmaS_odd[f"_{k}{i}{j}"] = sp.symbols(f"σS_odd_{k}{i}{j}")
                sigmaS_even[f"_{k}{i}{j}"] = sp.symbols(f"σS_even_{k}{i}{j}")
    
    return sigmaC_odd, sigmaC_even, sigmaS_odd, sigmaS_even


def symbolic_constraints(R_space, R_spin, f=1, noso=False):
    """
    计算电导(CC)和自旋电导(SC)的约束关系
    参数:
        R_space: 空间变换矩阵 (3x3 矩阵)
        R_spin: 自旋变换矩阵 (3x3 矩阵)
        f: 时间反演时为 -1，否则为 1
    返回:
        solutions: 符号变量的解
    """
    def trans_component(i, j, R_space, sigma, f=1, k=None, R_spin=None, is_odd=True, noso=False):
        """
        SHC单分量变换函数: σ_{ijk}' = Rᵢₘ Rⱼₙ (Rₛ)ₖₚ σ_{mnp}
        HC单分量变换函数: σ_{ij}' = Rᵢₘ Rⱼₙ σ_{mn}
        其中:
            - i, j, k: 分量索引
            - R_space: 空间变换矩阵 (3x3 矩阵)
            - R_spin: 自旋变换矩阵 (3x3 矩阵)
            - sigma: SHC或HC张量的符号变量字典
            - is_odd: 是否为时间反演奇对称
            - f: 时间反演时为 -1，否则为 1
        Returns:
            trans_sigma_component: 变换后的单分量
        """
        trans_sigma_component = 0
        char = ['x', 'y', 'z']
        Char = ['X', 'Y', 'Z']

        for m in range(3):  # m: 变换前电流分量索引
            for n in range(3):  # n: 变换前电场分量索引
                if k is not None and R_spin is not None:  # 有自旋分量的情况
                    for l in range(3):  # l: 变换前自旋分量索引
                        if is_odd:
                            if noso:
                                # 自旋群下的奇自旋张量变换
                                coef = R_spin[k, l]* R_space[i, m] * R_space[j, n]
                            else:
                                # 磁群下的奇自旋张量变换
                                coef = f * R_space.det() * R_space[k, l] * R_space[i, m] * R_space[j, n]
                        else:
                            if noso:
                                # 自旋群下的偶自旋张量变换
                                coef = R_spin.det() * R_spin[k, l] * R_space[i, m] * R_space[j, n]
                            else:
                                # 磁群下的偶自旋张量变换
                                coef = R_space.det() * R_space[k, l] * R_space[i, m] * R_space[j, n]
                        # 只考虑非零系数的项
                        if coef != 0:
                            trans_sigma_component += coef * sigma[f"_{Char[l]}{char[m]}{char[n]}"]
                else:
                    if is_odd:
                        # 磁点群奇电荷张量变换
                        coef = f * R_space[i, m] * R_space[j, n]
                    else:
                        # 偶电荷张量变换
                        coef = R_space[i, m] * R_space[j, n]
                    if coef != 0:
                        trans_sigma_component += coef * sigma[f"{char[m]}{char[n]}"]
        return trans_sigma_component
    

    def whether_identify(a,b):
        # 检查是否是恒等式：a == b
        is_identity = False
        if b.is_Mul and len(b.args) == 2:
            # 检查是否是系数乘以a的形式
            if b.args[1] == a:
                coef = b.args[0]
                # 检查系数是否接近1
                if isinstance(coef, (int, sp.Integer)):
                    is_identity = (coef == 1)
                elif hasattr(coef, 'evalf'):
                    is_identity = abs(coef.evalf() - 1) < 1e-9
        elif a == b:
            is_identity = True
        return is_identity

    def add_constraints(constraints, sigma, R_space, f, R_spin=None, is_odd=True):
        """
        添加约束条件到约束列表中
        参数:
            constraints: 约束列表
            sigma: SHC或HC张量的符号变量字典
            R_space: 空间变换矩阵 (3x3 矩阵)
            R_spin: 自旋变换矩阵 (3x3 矩阵)
            f: 时间反演时为 -1，否则为 1
            is_odd: 是否为时间反演奇对称
        返回:
            constraints: 更新后的约束列表
        """
        for i_idx, i_sym in enumerate(['x','y','z']):
            for j_idx, j_sym in enumerate(['x','y','z']):
                if R_spin is not None: # 有自旋分量的情况
                    for k_idx, k_sym in enumerate(['X','Y','Z']):
                        sigma_component = sigma[f"_{k_sym}{i_sym}{j_sym}"]
                        trans_sigma_component = trans_component(i_idx, j_idx, 
                            R_space, sigma, f=f, k=k_idx, R_spin=R_spin, is_odd=is_odd, noso=noso)
                        
                        # 只添加非恒等式的约束条件
                        is_identity = whether_identify(sigma_component, trans_sigma_component)
                        if not is_identity:
                            constraints.append(sp.Eq(sigma_component, trans_sigma_component))
                
                else: # 没有自旋分量的情况
                    sigma_component = sigma[f"{i_sym}{j_sym}"]
                    trans_sigma_component = trans_component(i_idx, j_idx, 
                            R_space, sigma, f=f, is_odd=is_odd, noso=noso)
                    
                    # 只添加非恒等式的约束条件
                    is_identity = whether_identify(sigma_component, trans_sigma_component)
                    if not is_identity:
                        constraints.append(sp.Eq(sigma_component, trans_sigma_component))
        return constraints
    

    def self_constraints(constraints, sigmaC_odd, sigmaC_even):
        # 添加时间反演奇对称的额外约束
        constraints.append(sp.Eq(sigmaC_odd['xy'], -sigmaC_odd['yx']))
        constraints.append(sp.Eq(sigmaC_odd['xz'], -sigmaC_odd['zx']))
        constraints.append(sp.Eq(sigmaC_odd['yz'], -sigmaC_odd['zy']))
        constraints.append(sp.Eq(sigmaC_odd['xx'], 0))
        constraints.append(sp.Eq(sigmaC_odd['yy'], 0))
        constraints.append(sp.Eq(sigmaC_odd['zz'], 0))

        # 添加时间反演偶对称的额外约束
        constraints.append(sp.Eq(sigmaC_even['xy'], sigmaC_even['yx']))
        constraints.append(sp.Eq(sigmaC_even['xz'], sigmaC_even['zx']))
        constraints.append(sp.Eq(sigmaC_even['yz'], sigmaC_even['zy']))
        
        return constraints


    # 定义符号变量
    sigmaC_odd, sigmaC_even, sigmaS_odd, sigmaS_even = define_symbolic_variables()
    
    # 添加电导(CC)约束
    constraints = add_constraints([], sigmaC_odd, R_space, \
        f, R_spin=None, is_odd=True)
    constraints = add_constraints(constraints, sigmaC_even, R_space, \
        f, R_spin=None, is_odd=False)
    constraints = self_constraints(constraints, sigmaC_odd, sigmaC_even)

    # 添加自旋电导(SC)约束
    constraints = add_constraints(constraints, sigmaS_odd, R_space, \
        f, R_spin=R_spin, is_odd=True)
    constraints = add_constraints(constraints, sigmaS_even, R_space, \
        f, R_spin=R_spin, is_odd=False)

    # 收集所有变量，求解约束方程
    all_symbols = list(sigmaC_odd.values()) + list(sigmaC_even.values()) \
                + list(sigmaS_odd.values()) + list(sigmaS_even.values())

    # 求解线性方程组
    solutions = sp.solve(constraints, all_symbols, dict=True)
    return solutions


def generators(Generators, R_spin, R_space, f):
    """
    判断一个空间变换和自旋变换是否是生成元
    参数:
        Generators: 生成元列表，每个元素为[R_spin, R_space, f]
        R_spin: 自旋变换矩阵 (3x3 矩阵)
        R_space: 空间变换矩阵 (3x3 矩阵)
        f: 时间反演时为 -1，否则为 1
    返回:
        is_generator: 是否为生成元
        Generators: 更新后的生成元列表
    """
    if Generators == []:
        Generators.append([R_spin, R_space,f])
        is_generator = True
    elif [R_spin, R_space,f] not in Generators:
        # 判断是否为生成元
        is_generator = True
        for i in range(len(Generators)):
            for j in range(len(Generators)):
                if is_generator:
                    Gen_spin = Generators[i][0] * Generators[j][0]
                    Gen_space = Generators[i][1] * Generators[j][1]
                    Gen_f = Generators[i][2] * Generators[j][2]
                    if Gen_spin == R_spin and Gen_space == R_space and Gen_f == f:
                        print(f"R_spin = {Generators[i][0]}*{Generators[j][0]},\
                            \nR_space = {Generators[i][1]}*{Generators[j][1]},\
                            \n是第{i+1}个生成元操作和第{j+1}个生成元操作的乘积，故不是生成元")
                        is_generator = False
        # 如果是生成元，添加到生成元列表
        if is_generator:
            Generators.append([R_spin, R_space,f])
    elif [R_spin, R_space,f] in Generators:
        print("该操作是重复的\n")
        is_generator = False

    return is_generator


def print_constraints(solutions, tot_constraint_exprs):
    """
    打印约束关系
    参数:
        solutions: 求解得到的解列表
        tot_constraint_exprs: 所有约束表达式的列表
    """
    def simplify_coefficients(expr):
        """简化表达式中的系数，将接近整数的分数简化为整数"""
        if expr.is_Number:
            return Gs.precise_num(expr)
        elif expr.is_Mul:
            # 分解为系数和变量部分
            coeff, vars_part = expr.as_coeff_mul()
            # 简化系数
            return Gs.precise_num(coeff) * sp.Mul(*vars_part)
        elif expr.is_Add:
            # 对加法表达式中的每个项进行简化
            return sp.Add(*[simplify_coefficients(term) for term in expr.args])
        elif expr.is_Pow:
            # 对指数表达式的底数和指数进行简化
            return sp.Pow(simplify_coefficients(expr.base), 
                         simplify_coefficients(expr.exp))
        else:
            return expr

    def print_each(solutions, prefix, prefix_ch, tot_constraint_exprs):
        print(f"\n时间反演{prefix_ch}约束关系:")
        ready_print = []
        for symbol, sym_obj in solutions[0].items():
            if str(symbol).startswith(prefix):
                ready_print.append(f"{symbol} = {simplify_coefficients(sym_obj)}")
        if ready_print == []:
            print("\t没有多余的解")
        else:
            for i in ready_print:
                if i not in tot_constraint_exprs:
                    tot_constraint_exprs.append(i)
                    print("\t",i)
                else:
                    print("\t",i,"\t该约束关系已在之前的操作结果中存在")
        return tot_constraint_exprs

    # 输出时间反演奇约束关系
    tot_constraint_exprs = print_each(solutions, "σC_odd", "奇电导", tot_constraint_exprs)
    tot_constraint_exprs = print_each(solutions, "σS_odd", "奇自旋电导", tot_constraint_exprs)
    
    # 输出时间反演偶约束关系
    tot_constraint_exprs = print_each(solutions, "σC_even", "偶电导", tot_constraint_exprs)
    tot_constraint_exprs = print_each(solutions, "σS_even", "偶自旋电导", tot_constraint_exprs)
    return tot_constraint_exprs


def main():
    args = parse_args()
    input_file, noso = (
        args.input_file,
        args.noso,
    )

    # 读取磁空间群与自旋空间群的标准操作
    if noso:
        R_spins, R_spaces, R_spins_Frac, R_spaces_Frac, translates =\
            Gs.grab_from_spin_space_group(input_file)
        fs = [R_spin.det() for R_spin in R_spins]
    else:
        R_spins_Frac, R_spaces_Frac, R_spins, R_spaces, fs =\
             get_MPG_operators(input_file)
        translates = sp.zeros(len(R_spins),1)
    
    # 对每个标准操作进行约束条件计算
    j, k = 1, 1
    Generators = [] # 空间变换和自旋变换的生成元
    tot_constraint_exprs = []

    for R_spin, R_space, R_spin_F, R_space_F, f, translate \
        in zip(R_spins, R_spaces, R_spins_Frac, R_spaces_Frac, fs, translates):
        R_space = sp.Matrix(R_space)
        R_spin = sp.Matrix(R_spin)
        print(f"第{j}个操作:")
        if noso:
            print(f"\tR_spin_Frac is {R_spin_F}\n\tR_space_Frac is {R_space_F}\
                \n\tR_spin_Cart is {R_spin}\n\tR_space_Cart is {R_space}\
                \n\tTranslate is {translate}")
        else:
            print(f"\tR_spin_Frac is {R_spin_F}\n\tR_space_Frac is {R_space_F}\
                \n\tR_spin_Cart is {R_spin}\n\tR_space_Cart is {R_space}\
                \n\tTime reversal is {f}")

        # 判断是否为生成元
        is_generator = generators(Generators, R_spin, R_space, f)
        # 如果是生成元，计算约束条件
        if is_generator:
            print(f"第{k}个生成元操作")
            # 计算约束条件
            solutions = symbolic_constraints(R_space, R_spin, f, noso=noso)
            # 输出约束关系
            tot_constraint_exprs = print_constraints(solutions, tot_constraint_exprs)
            k += 1
        j += 1
        print(80*"=","\n")
    
    # 对所有操作的约束求解
    print("总约束条件")
    sigmaC_odd, sigmaC_even, sigmaS_odd, sigmaS_even = define_symbolic_variables()
    all_symbols = list(sigmaC_odd.values()) + list(sigmaC_even.values()) \
                + list(sigmaS_odd.values()) + list(sigmaS_even.values())

    # 将tot_constraint_exprs转换为方程形式
    tot_constraint_eqs = []
    for cons_expr in tot_constraint_exprs:
        left, right = cons_expr.split('=')
        tot_constraint_eqs.append(sp.Eq(sp.simplify(left), sp.simplify(right)))

    tot_solutions = sp.solve(tot_constraint_eqs, all_symbols, dict=True)
    print_constraints(tot_solutions, [])
    print(80*"=")
    

if __name__ == "__main__":
    main()