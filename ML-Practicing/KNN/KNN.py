import numpy as np 
import operator 

"""
inX 待测试的数据
dataSet 已经分好类的数据集
"""


def classify(inX , dataSet , labels , k) :
    dataSetSize = dataSet.shape[0] # dataSet 行数
    # 行向量方向上重复 inX dataSetSize ，列方向重复 1 次
    diffMat = np.tile(inX , (dataSetSize , 1)) - dataSet 
    
    # 二维特征相减后平方
    sqDiffMat = diffMat ** 2 
    
    # sum 函数，axis = 0 按照列相加 , axis = 1 按照行相加
    sqDistances = sqDiffMat.sum(axis = 1) 
    
    # 开方，计算出距离
    distances = sqDistances ** 0.5

    # 返回 distances 中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort() 

    #定一个记录类别次数的字典
    classCount = {} 
    for i in range(k) : 
        # 取出前 k 个元素的类别
        voteLabel = labels[sortedDistIndices[i]]
        # dict.get(key , default = None), 字典的 get() 方法，返回指定键的值，如果值不在字典中返回默认值，计算类别次数
        classCount[voteLabel] = classCount.get(voteLabel , 0) + 1;

    # 根据字典中的值 降序 处理
    sortedClassCount = sorted(classCount.items() , key = operator.itemgetter(1) , reverse=True) 
    # 返回次数最多的类别，即所要分类的类别
    
    return sortedClassCount[0][0]  

def createDataSet() :
    group = np.array([[1,101] , [5,89] , [108,5] , [115, 8]])
    labels = ['爱情片' , '爱情片' , '动作片' , '动作片'] 
    return group , labels 

if __name__ == '__main__' :
    # 创建数据集
    group , labels = createDataSet()  
    
    # 测试集
    test = [101 , 20] 

    # KNN 分类
    test_class = classify(test , group , labels , 3)

    # 打印分类结果
    print(test_class)