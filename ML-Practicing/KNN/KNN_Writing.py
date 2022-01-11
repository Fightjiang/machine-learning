import numpy as np 
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN 

def img2vector(filename) :
    returnVet = np.zeros((1,1024))
    fr = open(filename)

    for i in range(32) :
        lineStr = fr.readline() 
       #print(lineStr)
        for j in range(32) :
            returnVet[0 , i*32 + j] = int(lineStr[j])
    
    return returnVet 

def handWritingClassTest() :
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m , 1024))
    
    hwLabels = []
    for i in range(m) :
        fileNameStr = trainingFileList[i] 
        classNumber = int(fileNameStr.split("_")[0])
        hwLabels.append(classNumber) 
        trainingMat[i] = img2vector('trainingDigits/%s' % (fileNameStr))

    neigh = KNN(n_neighbors = 3 , algorithm = 'auto')
    neigh.fit(trainingMat , hwLabels) 

    testFileList = listdir('testDigits') 

    errorCount = 0.0
    for i in range(len(testFileList)) :
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split("_")[0])
        vectorTest = img2vector('testDigits/%s' % fileNameStr) 
        PreResult = neigh.predict(vectorTest)
        print("预测数字结果：%d \t 真实的数字:%d" % (PreResult , classNumber))
        if(PreResult != classNumber) :
            ++errorCount

    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/len(testFileList) * 100))

if __name__ == '__main__' :
    handWritingClassTest() ; 