import matplotlib.pyplot as plt
import numpy
import numpy as np
import random
from loadData import loadData
from copy import deepcopy
import os

class main:

    def np_arr_to_list(self, np_arr):
        result = []
        for i in np_arr:
            result.append(int(i))
        return result

    def prepare_data(self):
        matrix = []
        for i in range(len(self.dataAll[b'data'])):
            row = self.np_arr_to_list(self.dataAll[b'data'][i])
            row.append(self.dataAll[b'labels'][i])
            row.append(i)
            matrix.append(row)
        print(len(matrix))
        return matrix

    def prepare_data2(self):
        matrix = []
        for i in range(len(self.dataAll)):
            batch =self.dataAll[i]
            for j in range(len(batch[b'data'])):
                row = self.np_arr_to_list(batch[b'data'][j])
                row.append(batch[b'labels'][j])
                row.append(j)
                matrix.append(row)
        print(len(matrix))
        return matrix

    def samples(self,cluster):
        sampleList=[]
        for i in range(min(20,len(cluster))):
            arr=cluster[i]
            sampleList.append(arr[:3072])

        return sampleList


    def max_in_each_cluster(self,cluster):
        frequencyArray=[0 for i in range(10)]
        for array in cluster:
            label=array[self.labelIndex]
            frequencyArray[label]+=1
        maximIndex=frequencyArray.index(max(frequencyArray))
        maxim=max(frequencyArray)
        percentage=maxim*100/len(cluster)
        return maximIndex,percentage


    def __init__(self):
        self.labelIndex=3072
        self.categories={
                0: "airplane",
                1: "automobile",
                2: "bird",
                3: "cat",
                4: "deer",
                5: "dog",
                6: "frog",
                7: "horse",
                8: "ship",
                9: "truck"
        }
        self.DATA_SIZE = 3072

        self.data1=loadData.unpickle('./data_batch_1')
        self.data2=loadData.unpickle('./data_batch_2')
        self.data3=loadData.unpickle('./data_batch_3')
        self.data4=loadData.unpickle('./data_batch_4')
        self.data5=loadData.unpickle('./data_batch_5')
        self.test=loadData.unpickle('./test_batch')
        #choose the batches you want
        self.dataAll=[self.data1,self.data2,self.data3,self.data4,self.data5]
        matrix=self.prepare_data2()

        #self.dataAll = loadData.unpickle('./data_batch_1')
        #matrix = self.prepare_data()

        #choose number of clusters you want and iterations
        self.k = 2
        self.it = 2


        centeroidList = []
        graph = []
        self.kTemp = self.k
        iterations = 0
        # generating random centeroids for the beginning only.
        while self.k > 0:
            c = random.randint(0, len(self.dataAll[0][b'data']))
            centeroid = self.dataAll[0][b'data'][c]
            # self.create_folders(dataAll, c)
            centeroidList.append(centeroid)
            self.k -= 1


        while iterations < self.it:
            clusterDistortion = 0
            allDistortion = 0
            clusters = [[] for i in range(len(centeroidList))]
            temp = numpy.array(centeroidList, dtype="uint8")
            #for centeroid in temp:
             #  self.image(centeroid)
            imageIndex = 0
            for row in matrix:
                minDistance = 1e10
                minCenteroidIndex = 0
                for index in range(len(centeroidList)):
                    centeroid = centeroidList[index]
                    distance = self.calculateDistance(row, centeroid)
                    if (distance < minDistance):
                        minDistance = distance
                        minCenteroidIndex = index

                clusters[minCenteroidIndex].append(row)
                imageIndex += 1


            for index in range(len(clusters)):
                centeroid = centeroidList[index]
                clusterDistortion =0
                for array in clusters[index]:
                    currentDifference =  (
                        int(self.calculateDistortionMeasure(array, centeroid)))
                    clusterDistortion = int(clusterDistortion) + currentDifference
                allDistortion = int(allDistortion) + int(clusterDistortion) ** 2

            centeroidList.clear()

            for cluster in clusters:
                centeroidList.append(self.calculateNewCenteroid(cluster))

            t = (iterations, allDistortion)
            graph.append(t)
            print("distortion in iteration ",iterations,"is ", allDistortion)

            iterations += 1
        self.graph(graph)
        for index in range(len(clusters)):
            cluster=clusters[index]
            centeroid=centeroidList[index]
            clusterAccuracy,perc=self.max_in_each_cluster(cluster)
            sampleList=self.samples(cluster)
            temp = numpy.array(sampleList, dtype="uint8")
            #for sample in temp:
                #self.image(sample)
            print("accuracy",perc)
            centeroidImageNumpy = numpy.array(centeroid, dtype="uint8")
            centeroidImage=self.image(centeroidImageNumpy)
            #self.save_data(centeroidImage,cluster,index,temp,perc)


    def save_data(self,centeroidImage, cluster, centroid_id,temp,percentage):
        freq,percentage = self.max_in_each_cluster(cluster)
        cluster_path = f'./samples/clusterNum={centroid_id} size={len(cluster)} {self.categories[freq]}'
        os.makedirs(cluster_path, 0o666, True)
        plt.imsave(cluster_path + f'/centroid{centroid_id}.png', centeroidImage)

        for index in range(len(temp)):
            sample=temp[index]
            image=self.image(sample)
            plt.imsave(cluster_path + f'/img-{index}.png',image)
        f = open(cluster_path+f'/demofile2.txt', "a")
        str=f'accuracy is: {percentage} %\n number of k clusters are: {self.kTemp}\n number of iterations are: {self.it}'
        f.write(str)
        f.close()



    def image(self, img):
        im_r = img[0:1024].reshape(32, 32)
        im_g = img[1024:2048].reshape(32, 32)
        im_b = img[2048:].reshape(32, 32)
        img = numpy.dstack((im_r, im_g, im_b))
        plt.imshow(img)
        plt.show()
        return img

    def graph(self, distortions):
        x = []
        y = []
        for i in range(len(distortions)):
            x.append(distortions[i][0])
            y.append(distortions[i][1])
        plt.plot(x, y)
        plt.xlabel('iterations')
        plt.ylabel('distortions')
        plt.title('distortion_graph')
        plt.savefig("distortion graph")
        plt.show()


    def calculateDistortionMeasure(self, array, centeroid):
        totalDistance = 0
        for index in range(self.DATA_SIZE):
            totalDistance += abs((int(array[index]) - int(centeroid[index])))
        return totalDistance

    # given array and centeroid ..calculate mean distance between them
    def calculateDistance(self, dataArray, centeroidArray):
        distance = 0
        for i in range(len(dataArray) - 2):
            distance += abs(int(dataArray[i]) - int(centeroidArray[i]))
        return distance

    # given cluster (list of array) and return new centeroid for this cluster
    def calculateNewCenteroid(self, cluster):
        totalArrays = len(cluster)
        newCenteroid = [0 for i in range(3072)]
        for array in cluster:
            for index in range(len(array)-2):
                newCenteroid[index] += array[index]
        for index in range(len(newCenteroid)):
            newCenteroid[index] = newCenteroid[index] / totalArrays
        return newCenteroid


m = main()
