from asyncio import base_futures
from distutils.log import info
from email.charset import BASE64
from email.mime import base
from enum import unique
from math import log
from venv import create 

def createDataSet() :
    dataSet = [[0, 0, 0, 0, 'no'],         #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']] 
    labels = ['年龄' , '有工作' , '有自己的房子' , '信贷情况'] 
    return dataSet , labels 

# 计算 香农熵
def calcShannonEnt(dataSet) :
    numEntires = len(dataSet) 
    labelCounts = {} 
    for featVec in dataSet :
        label = featVec[-1] 
        labelCounts[label] = labelCounts.get(label , 0) + 1  
    
    shannonEnt = 0.0  
    for key in labelCounts : # yes or No 两种
        prob = float(labelCounts[key]) / numEntires
        shannonEnt -= prob * log(prob , 2) 

    return shannonEnt 

def splitDataSet(dataSet , axis , feature) :
    retDataSet = [] 
    for dataLine in dataSet:
        if dataLine[axis] == feature:
            # 去掉 axis 特征，将符合条件的添加到返回的数据集中
            # reducedFeatVec = dataLine[:axis] 
            #reducedFeatVec.extend(dataLine[axis + 1 :])
            # retDataSet.append(reducedFeatVec) 
            retDataSet.append(dataLine)

    return retDataSet

def chooseBestFeatureToSplit(dataSet , features) :
    numFeatures = len(features)
    baseEntropy = calcShannonEnt(dataSet) # 整个数据集的香农熵 
    bestInfoGain = 0.0  # 最优的信息增益
    bestFeature = -1    # 最优特征的索引值

    for i in range(numFeatures) :
        # 将 dataSet 中的数据先按行依次放入 example 中，然后取得 example 中的第 i 列的元素的集合，放入列表 featList 中
        featList = [example[i] for example in dataSet]
        uniqueFeature = set(featList) 
        tmpEntorpy = 0.0
        for feature in uniqueFeature :
            subDataSet = splitDataSet(dataSet , i , feature) 
            prob = float(len(subDataSet)) / len(dataSet)
            tmpEntorpy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - tmpEntorpy
        # print("%f %f %f " % (baseEntropy , tmpEntorpy , infoGain)) 
        print("第 %d 个特征的信息%s的增益为 %.3f " % (i , features[i] ,  infoGain)) 
        if infoGain > bestInfoGain :   
            bestInfoGain = infoGain 
            bestFeature = i 
    
    return bestFeature
    


if __name__ == "__main__" :
    dataSet , features = createDataSet() 
    # print(dataSet) 
    # print(calcShannonEnt(dataSet)) 
    print("最优特征索引值：" + str(features[chooseBestFeatureToSplit(dataSet , features)])) ; 