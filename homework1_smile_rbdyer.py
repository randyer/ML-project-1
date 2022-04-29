import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

#function (percentage correct)
def fPC (y, yhat):
    return np.mean(y == yhat)

# set of predictors, set of images, ground truth labels of set
def measureAccuracyOfPredictors (predictors, X, y):
    #empty vector to store guesses
    guess = np.zeros(len(y))

    #loop through each feature
    for feature in predictors:
        r1, c1, r2, c2 = feature
        #if positive number prediction is smiling
        smilepredictor = X[:,r1,c1] - X[:,r2,c2]
        smilepredictor[smilepredictor > 0] = 1
        smilepredictor[smilepredictor <= 0] = 0
        guess += smilepredictor

    mean = guess/len(predictors)
    finalPrediction = mean > 0.5#mean[mean > 0.5] = 1
    #returns accuracy of prediction
    return fPC(y, finalPrediction)

def stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels):

    predictors = [] #list of the best features (tuples)
    bestPredictorPercent = 0

# loop through every image and calculate the acccuracy of feature
    for i in range(6):
        bestPredictor = 0
        bestFeature = None

        for r1 in range(24):
            for c1 in range(24):
                for r2 in range(24):
                    for c2 in range(24):
                        #if feature is already in predictions list, skip it
                        if (r1,c1,r2,c2) in predictors:
                            continue
                        # if pixel 1 is pixel 2, skip comparison
                        if (r1,c1) == (r2,c2):
                            continue

                        #use measureAccuracyOfPredictors to find the acurracy of predictors
                        currentPrediction = measureAccuracyOfPredictors(predictors + list(((r1,c1,r2,c2),)),trainingFaces,trainingLabels)

                        if currentPrediction > bestPredictor:
                            bestPredictor = currentPrediction
                            bestFeature = (r1,c1,r2,c2)
        print(bestFeature)
        predictors.append(bestFeature)

    show = True
    if show:
        r1, c1 ,r2, c2 = predictors[0]
        r3, c3, r4, c4 = predictors[1]
        r5, c5, r6, c6 = predictors[2]
        r7, c7 ,r8, c8 = predictors[3]
        r9, c9, r10, c10 = predictors[4]
        r11, c11, r12, c12 = predictors[5]

        print(r1,c1,r2,c2,r3,c3,r4,c4,r5,c5,r6,c6,r7,c7,r8,c8,r9,c9,r10,c10,r11,c11,r12,c12)
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # Show r1,c1
        rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Show r3,c3
        rect = patches.Rectangle((c3 - 0.5, r3 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r4,c4
        rect = patches.Rectangle((c4 - 0.5, r4 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Show r5,c5
        rect = patches.Rectangle((c5 - 0.5, r5 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r6,c6
        rect = patches.Rectangle((c6 - 0.5, r6 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Show r6,c6
        rect = patches.Rectangle((c7 - 0.5, r7 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r6,c6
        rect = patches.Rectangle((c8 - 0.5, r8 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Show r6,c6
        rect = patches.Rectangle((c9 - 0.5, r9 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r6,c6
        rect = patches.Rectangle((c10 - 0.5, r10 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Show r6,c6
        rect = patches.Rectangle((c11 - 0.5, r11 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r6,c6
        rect = patches.Rectangle((c12 - 0.5, r12 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Display the merged result
        plt.show()
    print('training accuracy of size ',len(trainingLabels), ': ',measureAccuracyOfPredictors(predictors,trainingFaces,trainingLabels))
    print('testing accuracy of size ', len(testingLabels), ': ',measureAccuracyOfPredictors(predictors,testingFaces,testingLabels))
    return predictors



def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    testList = [400,600,800,1000,1200,1400,1600,1800,2000]
    for index,testSize in enumerate(testList):
        print(stepwiseRegression(trainingFaces[:testSize, :, :], trainingLabels[:testSize], testingFaces[:testSize,:,:], testingLabels[:testSize]))
        print('\n')



## testing variables
testFeature1 = (0,0,1,1)
testFeature2 = (2,2,3,3)
testFeature3 = (4,4,5,5)
testFeature4 = (6,6,7,7)
testFeature5 = (8,8,9,9)
testFeature6 = (10,10,12,12)
listOfFeaturesT = [testFeature1,testFeature2,testFeature3,testFeature4,testFeature5,testFeature6]


#--------------------------- test fPC method ---------------------------##groundT = np.array([1,0,1,0])
# groundT = np.array([1,0,1,0])
# perfectT = np.array([1,0,1,0])
# halfT = np.array([1,1,1,1])
# noneT = np.array([0,1,0,1])
# print(fPC(groundT,perfectT))
# print(fPC(groundT,halfT))
# print(fPC(groundT,noneT))

#print(measureAccuracyOfPredictors(listOfFeaturesT,))
