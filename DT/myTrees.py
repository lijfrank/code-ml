#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dt 
@Author    @Version
-------    --------
Frank      1.0
@time   : 2020/4/25 13:35:14
@Desciption : None
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:09:34 2018
决策树ID3的实现
@author: weixw
"""
from math import log
import operator


# 原始数据
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]

    # no surfacing：不浮出水面是否可以生存
    #
    # flippers：是否有脚蹼

    feature_names = ['no surfacing', 'flippers']
    return dataSet, feature_names


# 多数表决器
# 列中相同值数量最多为结果
def majorityCnt(classList):
    classCounts = {}
    for value in classList:
        if value not in classCounts.keys():
            classCounts[value] = 0
        classCounts[value] += 1
    sortedClassCount = sorted(classCounts.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 划分数据集
# dataSet:原始数据集
# axis:进行分割的指定列索引
# value:指定列中的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featDataVal in dataSet:
        if featDataVal[axis] == value:
            # 下面两行去除某一项指定列的值，很巧妙有没有
            reducedFeatVal = featDataVal[:axis]
            reducedFeatVal.extend(featDataVal[axis + 1:])
            retDataSet.append(reducedFeatVal)
    return retDataSet


# 计算香农熵
def calcShannonEnt(dataSet):
    # 数据集总项数
    numEntries = len(dataSet)
    # 标签计数对象初始化
    labelCounts = {}
    for featDataVal in dataSet:
        # 获取数据集每一项的最后一列的标签值
        currentLabel = featDataVal[-1]
        # 如果当前标签不在标签存储对象里，则初始化，然后计数
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 熵初始化
    shannonEnt = 0.0
    # 遍历标签对象，求概率，计算熵
    for key in labelCounts.keys():
        prop = labelCounts[key] / float(numEntries)
        shannonEnt -= prop * log(prop, 2)
    return shannonEnt


# 选出最优特征列索引
def chooseBestFeatureToSplit(dataSet):
    # 计算特征个数，dataSet最后一列是标签属性，不是特征量
    numFeatures = len(dataSet[0]) - 1
    # 计算初始数据香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 初始化信息增益，最优划分特征列索引
    bestInfoGain = 0.0
    bestFeatureIndex = -1
    for i in range(numFeatures):
        # 获取每一列数据
        featList = [example[i] for example in dataSet]
        # 将每一列数据去重
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算条件概率
            prob = len(subDataSet) / float(len(dataSet))
            # 计算条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeatureIndex = i
    return bestFeatureIndex


# 决策树创建
def createTree(dataSet, labels):
    # 获取标签属性，dataSet最后一列，区别于labels标签名称
    classList = [example[-1] for example in dataSet]
    # 树极端终止条件判断
    # 标签属性值全部相同，返回标签属性第一项值
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 只有一个特征（1列）
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 获取最优特征列索引
    bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
    # 获取最优索引对应的标签名称
    bestFeatureLabel = labels[bestFeatureIndex]
    # 创建根节点
    myTree = {bestFeatureLabel: {}}
    # 去除最优索引对应的标签名，使labels标签能正确遍历
    del (labels[bestFeatureIndex])
    # 获取最优列
    bestFeature = [example[bestFeatureIndex] for example in dataSet]
    uniquesVals = set(bestFeature)
    for value in uniquesVals:
        # 子标签名称集合
        subLabels = labels[:]
        # 递归
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeatureIndex, value), subLabels)
    return myTree


# 获取分类结果
# inputTree:决策树字典
# featLabels:标签列表
# testVec:测试向量  例如：简单实例下某一路径 [1,1]  => yes（树干值组合，从根结点到叶子节点）
def classify(inputTree, featLabels, testVec):
    # 获取根结点名称，将dict转化为list
    firstSide = list(inputTree.keys())
    # 根结点名称String类型
    firstStr = firstSide[0]
    # 获取根结点对应的子节点
    secondDict = inputTree[firstStr]
    # 获取根结点名称在标签列表中对应的索引
    featIndex = featLabels.index(firstStr)
    # 由索引获取向量表中的对应值
    key = testVec[featIndex]
    # 获取树干向量后的对象
    valueOfFeat = secondDict[key]
    # 判断是子结点还是叶子节点：子结点就回调分类函数，叶子结点就是分类结果
    # if type(valueOfFeat).__name__=='dict': 等价 if isinstance(valueOfFeat, dict):
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


# 将决策树分类器存储在磁盘中，filename一般保存为txt格式
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb+')
    pickle.dump(inputTree, fw)
    fw.close()


# 将瓷盘中的对象加载出来，这里的filename就是上面函数中的txt文件
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)
