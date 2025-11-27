"""
Author: Jianting Dong (董建艇)
Date: 2025-11-25
Version: 2.3
功能：获取自旋空间群的标准操作，其中需要将分数坐标变换矩阵转换为笛卡尔坐标变换矩阵
"""

import sympy as sp
import json
import numpy as np

def transform_frac_to_cart(R_frac,lattice):
    """
    将分数坐标变换矩阵转换为笛卡尔坐标变换矩阵
    
    参数:
        R_frac: 分数坐标变换矩阵
        lattice: 晶格基矢 (笛卡尔坐标系)
    返回:
        R_cart: 笛卡尔坐标变换矩阵
    """
    return lattice.transpose() * R_frac * lattice.transpose().inv()


def read_json_data(json_file_path, json_key):
    """
    从json文件中读取对应字段的信息
    
    参数:
        json_file_path: JSON文件的路径
        json_key: 对应字段的键
    
    返回:
        json_data: 对应字段的信息
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if json_key in data:
            json_data = data[json_key]
            return json_data
        else:
            print(f"错误: JSON文件中未找到{json_key}字段")
            return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


def precise_num(num):
    num_pre = None
    sqrt_expressions = [
        (sp.sqrt(2),np.sqrt(2)),(sp.sqrt(2)/2,np.sqrt(2)/2),(sp.sqrt(2)/3,np.sqrt(2)/3),(sp.sqrt(2)/4,np.sqrt(2)/4),(sp.sqrt(2)/5,np.sqrt(2)/5),
        (sp.sqrt(2)/6,np.sqrt(2)/6),(sp.sqrt(2)/7,np.sqrt(2)/7),(sp.sqrt(2)/8,np.sqrt(2)/8),(sp.sqrt(2)/9,np.sqrt(2)/9),(sp.sqrt(2)/10,np.sqrt(2)/10),

        (sp.sqrt(3),np.sqrt(3)),(sp.sqrt(3)/2,np.sqrt(3)/2),(sp.sqrt(3)/3,np.sqrt(3)/3),(sp.sqrt(3)/4,np.sqrt(3)/4),(sp.sqrt(3)/5,np.sqrt(3)/5),
        (sp.sqrt(3)/6,np.sqrt(3)/6),(sp.sqrt(3)/7,np.sqrt(3)/7),(sp.sqrt(3)/8,np.sqrt(3)/8),(sp.sqrt(3)/9,np.sqrt(3)/9),(sp.sqrt(3)/10,np.sqrt(3)/10),

        (sp.sqrt(5),np.sqrt(5)),(sp.sqrt(5)/2,np.sqrt(5)/2),(sp.sqrt(5)/3,np.sqrt(5)/3),(sp.sqrt(5)/4,np.sqrt(5)/4),(sp.sqrt(5)/5,np.sqrt(5)/5),
        (sp.sqrt(5)/6,np.sqrt(5)/6),(sp.sqrt(5)/7,np.sqrt(5)/7),(sp.sqrt(5)/8,np.sqrt(5)/8),(sp.sqrt(5)/9,np.sqrt(5)/9),(sp.sqrt(5)/10,np.sqrt(5)/10),

        (sp.sqrt(6),np.sqrt(6)),(sp.sqrt(6)/2,np.sqrt(6)/2),(sp.sqrt(6)/3,np.sqrt(6)/3),(sp.sqrt(6)/4,np.sqrt(6)/4),(sp.sqrt(6)/5,np.sqrt(6)/5),
        (sp.sqrt(6)/6,np.sqrt(6)/6),(sp.sqrt(6)/7,np.sqrt(6)/7),(sp.sqrt(6)/8,np.sqrt(6)/8),(sp.sqrt(6)/9,np.sqrt(6)/9),(sp.sqrt(6)/10,np.sqrt(6)/10),

        (sp.sqrt(7),np.sqrt(7)),(sp.sqrt(7)/2,np.sqrt(7)/2),(sp.sqrt(7)/3,np.sqrt(7)/3),(sp.sqrt(7)/4,np.sqrt(7)/4),(sp.sqrt(7)/5,np.sqrt(7)/5),
        (sp.sqrt(7)/6,np.sqrt(7)/6),(sp.sqrt(7)/7,np.sqrt(7)/7),(sp.sqrt(7)/8,np.sqrt(7)/8),(sp.sqrt(7)/9,np.sqrt(7)/9),(sp.sqrt(7)/10,np.sqrt(7)/10),  

        (sp.sqrt(10),np.sqrt(10)),(sp.sqrt(10)/2,np.sqrt(10)/2),(sp.sqrt(10)/3,np.sqrt(10)/3),(sp.sqrt(10)/4,np.sqrt(10)/4),(sp.sqrt(10)/5,np.sqrt(10)/5),
        (sp.sqrt(10)/6,np.sqrt(10)/6),(sp.sqrt(10)/7,np.sqrt(10)/7),(sp.sqrt(10)/8,np.sqrt(10)/8),(sp.sqrt(10)/9,np.sqrt(10)/9),(sp.sqrt(10)/10,np.sqrt(10)/10),
        ]

    for expr, value in sqrt_expressions:
        if abs(num - value) < 1e-6:
            num_pre = expr
            break

    # 如果没有匹配到常见表达式，使用nsimplify
    if num_pre is None:
        num_pre = sp.nsimplify(num, tolerance=1e-6, rational=True, full=True)
    return num_pre


def precise_cell(lattice, transpose=False):
    mod = sp.sqrt(sum(x**2 for x in lattice[0]))
    Cell_mat = sp.Matrix([[precise_num(num/mod) for num in line] for line in lattice])

    if transpose:
        Cell_mat = Cell_mat.transpose()
    print("========= Standard Cell ==========")
    print(Cell_mat)
    return Cell_mat


def grab_from_spin_space_group(json_file_path):
    """
    从自旋空间群的json文件中读取对应字段的信息
    参数:
        json_file_path: JSON文件的路径
    返回:
        spin_operations: 自旋操作列表
        space_operations: 空间操作列表
    """
    # 读取Stadard_Cell
    std_Cell_data = read_json_data(json_file_path, "transformation_matrix_spin_cartesian_lattice_G0")
    std_Cell = precise_cell(std_Cell_data,transpose=True)
    
    # 读取G0_std_operations信息
    operations_data = read_json_data(json_file_path, "G0_std_operations_in_lattice")
    print(f"总操作数量: {len(operations_data)}")
    
    spin_operations_cart, space_operations_cart, translates = [], [], []
    spin_operations_frac, space_operations_frac = [], []

    # 处理操作数据
    for operation in operations_data:
        spin_operation_frac = sp.Matrix(operation[0]).applyfunc(precise_num)
        space_operation_frac = sp.Matrix(operation[1]).applyfunc(precise_num)
        spin_operations_frac.append(spin_operation_frac)
        space_operations_frac.append(space_operation_frac)
        translates.append(sp.Matrix(operation[2]).applyfunc(precise_num))
        
        # 将分数坐标变换矩阵转换为笛卡尔坐标变换矩阵
        spin_operation_cart = transform_frac_to_cart(spin_operation_frac, std_Cell)
        spin_operations_cart.append(spin_operation_cart.applyfunc(precise_num))
        space_operation_cart = transform_frac_to_cart(space_operation_frac, std_Cell)
        space_operations_cart.append(space_operation_cart.applyfunc(precise_num))
    print(80*"=","\n")

    return spin_operations_cart, space_operations_cart, \
            spin_operations_frac, space_operations_frac, translates