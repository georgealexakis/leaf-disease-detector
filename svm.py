import cv2 as cv
import numpy as np
import math
from skimage import feature, io
from sklearn import preprocessing
from scipy.stats import entropy, norm, kurtosis, skew
np.seterr(divide='ignore', invalid='ignore')


class SVM:

    def __init__(self):
        # Optimize OpenCV
        cv.useOptimized()

    def extractFeatures(self, dataset, num):
        trainingData = []
        labels = np.array([], dtype=np.int32)
        print("Feature extraction started.")
        print("Running will take a few minutes...")
        path = ""
        for i in range(num):
            # Choose right path and set label
            if (i < int(num/2)):
                counter = i
                path = dataset + "/healthy/img (" + str(i + 1) + ").JPG"
                labels = np.append(labels, -1)
            else:
                counter = i - int(num/2)
                path = dataset + "/rotten/img (" + str(counter + 1) + ").JPG"
                labels = np.append(labels, 1)
            # Display paths
            print(path)
            # Read dataset
            RGB = cv.imread(path)
            # Convert RGB image to chosen color space
            HSV = cv.cvtColor(RGB, cv.COLOR_BGR2HSV)
            # For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255]
            # Define color range
            lower = np.array([10, 100, 20])
            upper = np.array([20, 255, 200])
            # Threshold the HSV image to get only blue colors
            mask = cv.inRange(HSV, lower, upper)
            # Bitwise-AND mask and original image
            RGBmask = cv.bitwise_and(RGB, RGB, mask=mask)
            # RGB mask image to grayscale
            GI = cv.cvtColor(RGBmask, cv.COLOR_BGR2GRAY)
            # Feature extraction
            # GLCM features
            Grauwertmatrix = feature.greycomatrix(
                GI, distances=[1], angles=[0])
            Contrast = (feature.greycoprops(Grauwertmatrix, "contrast"))[0][0]
            Correlation = (feature.greycoprops(
                Grauwertmatrix, "correlation"))[0][0]
            Energy = (feature.greycoprops(Grauwertmatrix, "energy"))[0][0]
            Homogeneity = (feature.greycoprops(
                Grauwertmatrix, "homogeneity"))[0][0]
            # More features
            Mean = np.mean(RGBmask)
            STD = np.std(RGBmask)
            Entropy = entropy(RGBmask.flatten())
            if math.isnan(Entropy):
                Entropy = 0.0
            RMS = np.sqrt(np.mean(RGBmask**2))
            Variance = np.mean(np.var(RGBmask))
            Sum = np.sum(RGBmask)
            Smoothness = 1-(1/(1+Sum))
            Kurtosis = kurtosis(RGBmask.flatten())
            Skewness = skew(RGBmask.flatten())
            IDM = 0.0
            for i in range(GI.shape[0]):
                for j in range(GI.shape[1]):
                    temp = GI[i, j]/(1+(i-j)**2)
                    IDM = IDM + temp
            # Features to variable
            trainingData.append([Contrast, Correlation, Energy, Homogeneity,
                                 Mean, STD, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM])
        print("Feature extraction finished.")
        # Save data to file txt, npy
        np.savetxt(dataset + "_features.txt",
                   np.float32(trainingData), delimiter=",")
        np.save(dataset + "_features.npy", np.float32(trainingData))
        np.savetxt(dataset + "_labels.txt", labels, delimiter=",")
        np.save(dataset + "_labels.npy", labels)

    def imageSegmentation(self, path, label):
        # Dispaly messages
        print("Image segmentation started.")
        print("Running will take a few minutes...")
        # Init features array
        trainingData = []
        labels = np.array([], dtype=np.int32)
        # Optimize OpenCV
        cv.useOptimized()
        # Read images
        RGB = cv.imread(path)
        # Convert RGB image to chosen color space
        HSV = cv.cvtColor(RGB, cv.COLOR_BGR2HSV)
        # For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255]
        # Define color range
        lower = np.array([10, 100, 20])
        upper = np.array([20, 255, 200])
        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(HSV, lower, upper)
        # Bitwise-AND mask and original image
        RGBmask = cv.bitwise_and(RGB, RGB, mask=mask)
        # RGB mask image to grayscale
        GI = cv.cvtColor(RGBmask, cv.COLOR_BGR2GRAY)
        # Feature extraction
        # GLCM features
        Grauwertmatrix = feature.greycomatrix(GI, distances=[1], angles=[0])
        Contrast = (feature.greycoprops(Grauwertmatrix, "contrast"))[0][0]
        Correlation = (feature.greycoprops(
            Grauwertmatrix, "correlation"))[0][0]
        Energy = (feature.greycoprops(Grauwertmatrix, "energy"))[0][0]
        Homogeneity = (feature.greycoprops(
            Grauwertmatrix, "homogeneity"))[0][0]
        # More features
        Mean = np.mean(RGBmask)
        STD = np.std(RGBmask)
        Entropy = entropy(RGBmask.flatten())
        if math.isnan(Entropy):
            Entropy = 0
        RMS = np.sqrt(np.mean(RGBmask**2))
        Variance = np.mean(np.var(RGBmask))
        Sum = np.sum(RGBmask)
        Smoothness = 1-(1/(1+Sum))
        Kurtosis = kurtosis(RGBmask.flatten())
        Skewness = skew(RGBmask.flatten())
        IDM = 0.0
        for i in range(GI.shape[0]):
            for j in range(GI.shape[1]):
                temp = GI[i, j]/(1+(i-j)**2)
                IDM = IDM + temp
        # Features to array
        labels = np.append(labels, label)
        dataLabels = ["Contrast", "Correlation", "Energy", "Homogeneity", "Mean",
                      "STD", "Entropy", "RMS", "Variance", "Smoothness", "Kurtosis", "Skewness", "IDM"]
        trainingData.append([Contrast, Correlation, Energy, Homogeneity,
                             Mean, STD, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM])
        # Save data to file
        np.savetxt("single_image_features.txt",
                   np.float32(trainingData), delimiter=",")
        np.save("single_image_features.npy", np.float32(trainingData))
        np.savetxt("single_image_labels.txt", labels, delimiter=",")
        np.save("single_image_labels.npy", labels)
        # Display results
        cv.namedWindow("Original Image", cv.WINDOW_NORMAL)
        cv.resizeWindow("Original Image", 600, 600)
        cv.imshow("Original Image", RGB)
        cv.namedWindow("HSV Image", cv.WINDOW_NORMAL)
        cv.resizeWindow("HSV Image", 600, 600)
        cv.imshow("HSV Image", HSV)
        cv.namedWindow("Mask", cv.WINDOW_NORMAL)
        cv.resizeWindow("Mask", 600, 600)
        cv.imshow("Mask", mask)
        cv.namedWindow("RGB segmented part", cv.WINDOW_NORMAL)
        cv.resizeWindow("RGB segmented part", 600, 600)
        cv.imshow("RGB segmented part", RGBmask)
        cv.namedWindow("Grayscale segmented part", cv.WINDOW_NORMAL)
        cv.resizeWindow("Grayscale segmented part", 600, 600)
        cv.imshow("Grayscale segmented part", GI)
        # Print features
        for i in range(len(dataLabels)):
            print(dataLabels[i] + ": " + str(trainingData[0][i]))
        print("Label: " + str(labels[0]))
        # Wait to terminate windows
        cv.waitKey(0)

    def trainSVM(self, dataset):
        print("Training started.")
        print("Running...")
        # Load data
        data = np.load(dataset + "_features.npy")
        labels = np.load(dataset + "_labels.npy")
        # Train the SVM, Radial Basis Function (RBF) with a Gaussian Kernel
        svm = cv.ml.SVM_create()
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setKernel(cv.ml.SVM_RBF)
        svm.train(data, cv.ml.ROW_SAMPLE, labels)
        print("Training finished.")
        # Save model
        svm.save(dataset + "_model.xml")
        print("Model saved.")

    def trainSVMCustom(self, dataset):
        print("Training started.")
        print("Running...")
        # Load data
        data = np.load(dataset + "_features.npy")
        labels = np.load(dataset + "_labels.npy")
        # Train the SVM, Radial Basis Function (RBF) with a Gaussian Kernel
        svm = cv.ml.SVM_create()
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setKernel(cv.ml.SVM_RBF)
        svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 10000, 1e-7))
        svm.trainAuto(data, cv.ml.ROW_SAMPLE, labels)
        print("Training finished.")
        # Save model
        svm.save(dataset + "_model.xml")
        print("Model saved.")

    def predict(self, testingDataset, predictingDataset):
        # Load testing dataset
        data = np.load(testingDataset + "_features.npy")
        labels = np.load(testingDataset + "_labels.npy")
        # Prediction based on testing folder
        print("Prediction started.")
        # Load SVM model
        svm = cv.ml.SVM_load(predictingDataset + "_model.xml")
        predictions = svm.predict(data)[1]
        print(predictions)
        counter = 0
        accuracy = 0.0
        for i in range(len(predictions)):
            if (labels[i] == predictions[i][0]):
                counter = counter + 1
        # Print accuracy
        accuracy = (counter / len(labels)) * 100
        print("SVM model accuracy: " + str(accuracy) + " %")
        print("Prediction finished.")


if __name__ == '__main__':
    # New SVM model
    # ---- Classes ----
    # Healthy leaves  :  -1
    # Unhealthy leaves:   1
    svm = SVM()

    # Extract features -----> ONCE <----- and then use them to train the models or do predictions (uncomment below to run)
    # svm.extractFeatures("training_dataset_1", 2000)
    # svm.extractFeatures("training_dataset_2", 800)
    # svm.extractFeatures("testing_dataset", 40)

    # Train SVM model (uncomment below to run)
    # svm.trainSVM("training_dataset_1")
    # svm.trainSVMCustom("training_dataset_2")
    # svm.trainSVM("testing_dataset")

    # Start predictions put testing images and the trained model (uncomment below to run)
    # svm.predict("testing_dataset", "training_dataset_1")
    # svm.predict("testing_dataset", "training_dataset_2")
    # svm.predict("testing_dataset", "testing_dataset")

    # Put the link of the image to do segmentation of diseased part and extract features (uncomment below to run)
    svm.imageSegmentation("vine_leaves/img (1).jpg", 1)
    svm.predict("single_image", "training_dataset_1")
