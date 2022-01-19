from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator

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
            reducedFeatVec = dataLine[:axis] 
            reducedFeatVec.extend(dataLine[axis + 1 :])
            retDataSet.append(reducedFeatVec) 
            #retDataSet.append(dataLine)

    return retDataSet

def chooseBestFeatureToSplit(dataSet ) :
    numFeatures =  len(dataSet[0]) - 1
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
        # print("第 %d 个特征的信息的增益为 %.3f " % (i , infoGain)) 
        if infoGain > bestInfoGain :   
            bestInfoGain = infoGain 
            bestFeature = i 
    
    return bestFeature
    
# 统计 List 中出现最多的元素标签
def majorityCnt(classList) :
    classCount = {}
    for veto in classList :
        classCount[veto] += classCount.get(veto , 0) + 1 ;
    sortedClassCount = sorted(classCount.items() , key=operator.itemgetter(1) , reverse=True)
    return sortedClassCount[0][0] # 返回出现最多的元素

def createTree(dataSet , labels ) :
    classList = [example[-1] for example in dataSet] # 取得最终的分类标签
    # 如果完全相同的话，就停止划分，成为叶节点
    if classList.count(classList[0]) == len(classList) : 
        return classList[0]

    if len(dataSet[0]) == 1 : # 全部特征都用完了，还是不能将数据划分成唯一分类
        return majorityCnt(classList)
        
    bestFeat = chooseBestFeatureToSplit(dataSet) # 下标
    bestFeatLabel = labels[bestFeat] 
    myTree = {bestFeatLabel :{}} # 根据最优特征的标签生成树

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues) 
    for value in uniqueVals :
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet , bestFeat , value) , labels )
    return myTree 

# {'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
def getNumLeafs(myTree) :

    if(type(myTree).__name__ == 'str') :
        return 1 

    # python3 中 myTree.keys() 返回的是 dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，
    # 可以使用list(myTree.keys())[0] , 也可以使用以下方式
    firstStr = next(iter(myTree)) 
    secondDict = myTree[firstStr]
    
    numLeafs = 0
    for key in secondDict.keys():
        numLeafs += getNumLeafs(secondDict[key])
    return numLeafs

def getTreeDepth(myTree) :
    if type(myTree).__name__ == 'str' :
        return 1
    firstStr = next(iter(myTree)) 
    secondDict = myTree[firstStr]
    maxDepth = 0
    for key in secondDict.keys():
        maxDepth = max(maxDepth , getTreeDepth(secondDict[key]) + 1)
    return maxDepth 

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")                                            #定义箭头格式
    #font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)        #设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',    #绘制结点
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

"""
函数说明:标注有向边属性值
"""
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]                                            #计算标注位置                   
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

"""
函数说明:绘制决策树
"""
def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")        # 设置叶结点格式
    numLeafs = getNumLeafs(myTree)                      # 获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)                        # 获取决策树层数
    firstStr = next(iter(myTree))                       # 下个字典                                                 
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)    #中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)              # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制结点
    secondDict = myTree[firstStr]                       # 下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD # y偏移
    for key in secondDict.keys():                               
        if type(secondDict[key]).__name__=='dict':      # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))   # 不是叶结点，递归调用继续绘制
        else:                                           # 如果是叶结点，绘制叶结点，并标注有向边属性值                                             
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW 
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

"""
函数说明:创建绘制面板
"""
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white') #创建fig 
    plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号   
    fig.clf()            #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)     # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))                    # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))                   # 获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;      # x偏移
    plotTree(inTree, (0.5,1.0), '')                                 # 绘制决策树
    plt.show()                          

def classfy(myTree , features ,  testVec) :
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr] 
    featIndex = features.index(firstStr)
    
    for key in secondDict.keys():
        if testVec[featIndex] == key :
            if type(secondDict[key]).__name__ == 'dict' :
                return classfy(secondDict[key] , features, testVec)
            else :
                return secondDict[key]


if __name__ == "__main__" :
    dataSet , features = createDataSet() 
    # print(dataSet) 
    # print(calcShannonEnt(dataSet)) 
    # print("最优特征索引值：" + str(features[chooseBestFeatureToSplit(dataSet , features)])) ;

    myTree = createTree(dataSet , features) ; 
    print(myTree) 
    # createPlot(myTree)
    testVec = [0,0,0,1]
    result = classfy(myTree , features , testVec) 
    print(result) ; 