from cv2 import distanceTransform, mulSpectrums, norm
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines 
from matplotlib.font_manager import FontProperties
from numpy.lib.function_base import diff
import operator 

# 处理数据
def filedeal(filename) :
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines) 
    # 返回的 NumPy 矩阵，numberOfLines 行 ，3 列
    returnMat = np.zeros((numberOfLines , 3)) 

    # 返回的分类标签向量
    classLabelVector = [] 

    index = 0 
    for line in arrayLines :
        line = line.strip() 
        listFromLine = line.split('\t')
        for i in range(len(listFromLine) - 1):
            listFromLine[i] = float(listFromLine[i])

        returnMat[index] = listFromLine[0:3] 

        if listFromLine[-1] == 'didntLike' :
            classLabelVector.append(1) 
        elif listFromLine[-1] == 'smallDoses' :
            classLabelVector.append(2) 
        elif listFromLine[-1] == 'largeDoses' :
            classLabelVector.append(3)
        index += 1 

    return returnMat , classLabelVector

# 归一化数据
def autoNorm(dataSet) :
    minVal = dataSet.min(0)   # min(0) 每列的最小值 , 1 每行的最小值
    maxVal = dataSet.max(0) 
    range = maxVal - minVal  
    normDataSet = (dataSet - minVal) / range
    return normDataSet

# 可视化数据
def showdatas(datingDataMat , datingLabels) :
    # 将 fig 画布分隔成 1 行 1 列，不共享 x 轴和 y 轴，fig 画布的大小为（13，8）
    # 当 nrow=2,nclos=2 时，代表 fig 画布被划分为四个区域, ax[0][0] 表示第一行第一个区域
    axs = plt.subplots(nrows = 2 , ncols = 2 , sharex = False , sharey = False , figsize=(13,8))
    plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号
   
    LabelsColors = [] 
    for i in datingLabels :
        if i == 1 :
            LabelsColors.append("black") 
        elif i == 2 :
            LabelsColors.append("orange")
        elif i == 3 : 
            LabelsColors.append("red")

    axs[0][0].scatter(x = datingDataMat[:,0] , y = datingDataMat[:,1] , color=LabelsColors , s = 15 , alpha = .5)
    axs0_title_text  = axs[0][0].set_title("每年获得的飞行常客里程数与玩视频游戏所消耗时间占比")
    axs0_xlabel_text = axs[0][0].set_xlabel("每年获得的飞行常客里程数"  )
    axs0_ylabel_text = axs[0][0].set_ylabel("玩视频游戏所消耗时间占比"  )

    plt.setp(axs0_title_text , size = 9 , weight = 'bold' , color = 'red')
    plt.setp(axs0_xlabel_text , size = 7 , weight = 'bold' , color = 'black')
    plt.setp(axs0_ylabel_text , size = 7 , weight = 'bold' , color = 'black')

    axs[0][1].scatter(x = datingDataMat[:,0] , y = datingDataMat[:,2] , color=LabelsColors , s = 15 , alpha = .5)
    axs1_title_text  = axs[0][1].set_title("每年获得的飞行常客里程数与每周消费的冰淇淋公升占比")
    axs1_xlabel_text = axs[0][1].set_xlabel("每年获得的飞行常客里程数")
    axs1_ylabel_text = axs[0][1].set_ylabel("每周消费的冰淇淋公升")

    plt.setp(axs1_title_text , size = 9 , weight = 'bold' , color = 'red')
    plt.setp(axs1_xlabel_text , size = 7 , weight = 'bold' , color = 'black')
    plt.setp(axs1_ylabel_text , size = 7 , weight = 'bold' , color = 'black')

    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数' )
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比' )
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数' )
    plt.setp(axs2_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black') 


    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plt.show() 

def classify(inX , dataSet , labels , k) :
    diffMax = np.tile(inX , (dataSet.shape[0] , 1)) - dataSet 
    diffMax = diffMax ** 2 
    sqDistances = diffMax.sum(axis = 1)
    distances = sqDistances ** 0.5 

    sortedDisIndices = distances.argsort() 
    classCount = {} 
    for i in range(k) :
        Label = labels[sortedDisIndices[i]]
        classCount[Label] = classCount.get(Label , 0) + 1 

    sortedClassCount = sorted(classCount.items() , key = operator.itemgetter(1) , reverse=True)
    return sortedClassCount[0][0]

def datingClassTest() :
    filename = "datingTestSet.txt" 
    datingDataMat , datingLabels = filedeal(filename)
    normData = autoNorm(datingDataMat)
    numTestVecs = int(0.1 * datingDataMat.shape[0]) 
    errorCount = 0.0 

    for i in range(numTestVecs) :
        preResult = classify(normData[i] , normData[numTestVecs:] , datingLabels[numTestVecs:] , 4)
        print("预测分类结果：%d \t 真实的类别:%d" % (preResult , datingLabels[i]))
        if( preResult != datingLabels[i]) :
            errorCount += 1.0 

    print("错误率 :%f%%" % (errorCount / numTestVecs * 100))

if __name__ == '__main__': 
    '''
    listFromLine = "12 3214 432" 
    line = listFromLine.split()
    for i in range(len(line)):
        line[i] = float(line[i])
        print(type(line[i]))
    print(line) 
    
    filename = "datingTestSet.txt" 
    datingDataMat , datingLabels = filedeal(filename) 
    normData = autoNorm(datingDataMat)
    print(normData)
    #showdatas(datingDataMat , datingLabels)
    '''
    datingClassTest() 