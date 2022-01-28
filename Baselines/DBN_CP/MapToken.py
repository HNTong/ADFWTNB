
import javalang  # pip install javalang
import pandas as pd
import os
import numpy as np
import scipy.io as scio


SELECTED_NODES = (javalang.tree.MethodInvocation, javalang.tree.SuperMethodInvocation, javalang.tree.ClassCreator,
                     javalang.tree.PackageDeclaration, javalang.tree.InterfaceDeclaration,
                     javalang.tree.ClassDeclaration,
                     javalang.tree.ConstructorDeclaration, javalang.tree.MethodDeclaration,
                     javalang.tree.VariableDeclaration,
                     javalang.tree.FormalParameter, javalang.tree.IfStatement, javalang.tree.ForStatement,
                     javalang.tree.WhileStatement, javalang.tree.DoStatement, javalang.tree.AssertStatement,
                     javalang.tree.BreakStatement, javalang.tree.ContinueStatement, javalang.tree.ReturnStatement,
                     javalang.tree.ThrowStatement, javalang.tree.TryStatement, javalang.tree.SynchronizedStatement,
                     javalang.tree.SwitchStatement, javalang.tree.BlockStatement, javalang.tree.CatchClauseParameter,
                     javalang.tree.TryResource, javalang.tree.CatchClause, javalang.tree.SwitchStatementCase,
                     javalang.tree.ForControl, javalang.tree.EnhancedForControl, javalang.tree.BasicType,
                     javalang.tree.MemberReference, javalang.tree.ReferenceType, javalang.tree.SuperMemberReference,
                     javalang.tree.StatementExpression)


def index_type(node, SELECTED_NODES):
    i = 0
    for key in SELECTED_NODES:
        i += 1
        if isinstance(node, key):
            return i  # Forced end of for loop
    return 0


def mapping_token(data_path, save_path='C:/',
                  src_code_dir='E:/SourceCode/', runCount = ''):
    """
    :param data_path: e.g., 'E:/MORPH/CSV/ant-1.3.csv'
    :param save_path: e.g., 'C:/'
    :param src_code_dir: e.g., 'E:/Document/source file/'
    :return:
    """

    if len(data_path.split('/')) == 1:
        strs = data_path.split('\\')
    else:
        strs = data_path.split('/')
    data_name = strs[-1]
    data_name = data_name.replace('.csv', '')  # replace '.csv' with ''

    vector = []
    data = pd.read_csv(data_path)
    for i in range(data.shape[0]):  # each row/module
        str = data.iloc[i, 2]  # 3rd column which includes the information of modules' path
        str = str.replace(".", "/")  # replace '.' with '/'
        src_code_path = src_code_dir + data_name + '/src/java/' + str + '.java'

        temp = []
        if os.path.exists(src_code_path):  # Each module corresponds to an AST
            file = open(src_code_path, 'r')
            txt_java_code = file.read()
            tree_AST = javalang.parse.parse(txt_java_code)
            for path, node in tree_AST:
                mapping_value = index_type(node, SELECTED_NODES)  # self-defined function 'index_type'
                temp.append(mapping_value)

        if i == 0:
            vector = temp
        elif i == 1:
            vector = [vector, temp]
        else:
            vector.append(temp)

    if len(runCount) != 0:
            runCount = '_' + runCount
    scio.savemat(save_path + data_name + runCount + '.mat', {'token': vector})  # Must use dict, token is the variable name
