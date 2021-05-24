import os
import cv2
import numpy as np
import pickle5 as pickle
from sklearn.svm import SVC
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix  

class suppaluk():
    def __init__(self, train=False, 
                path="./src/model.sav", 
                img_size=(384, 512), 
                orientations=10, 
                pixels_per_cell=(32, 32), 
                cells_per_block=(3, 3)):
        '''
        Define HOG Parameter
        # 1 Block -> 9 Cell : 1 Cell -> 10 bins : 90 bins/block [x1, x2, ..., x90]
        '''
        self.orientations = orientations # Bins
        self.pixels_per_cell = pixels_per_cell # Number of pixels in cell
        self.cells_per_block = cells_per_block # Number of cell in block
        self.img_size = img_size # Img size

        # For train mode
        if not train:
            self.model = pickle.load(open(path, 'rb'))

    # Read image from byte data
    def readImgByte(self, image_byte):
        nparr = np.fromstring(image_byte, np.uint8)
        original_image = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        return original_image

    # Return image as blob (Byte steam)
    def showImgBlob(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, im_buf_arr = cv2.imencode(".jpg", img)
        byte_im = im_buf_arr.tobytes()
        return byte_im

    # Read image
    def readImg(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    # Show image
    def showImg(self, img):
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()
    
    # Extract feature by using HOG
    def extractFeature(self, img):
        # Resize imgage
        img_re = cv2.resize(img, self.img_size, interpolation = cv2.INTER_AREA)
        descriptor = hog(img_re, self.orientations, self.pixels_per_cell, self.cells_per_block, block_norm='L2', feature_vector=True)
        return descriptor

    # Draw text in picture
    def __drawText(self, img, text):
        # Define constat paprameter
        image_h, image_w, _ = img.shape # Get image width and height
        fontScale = image_h/400 # Front size
        thick = int(0.6 * (image_h + image_w) / 300) # Thickness
        color = (255,165,0) # Color // Orange
        c = 40 # Start point
        a = 10  # Adj. point

        t_size = cv2.getTextSize(text, 0, fontScale, thickness=thick // 2)[0] # Calculate text size
        start_point = (c+a,c+a)
        end_point = (c+t_size[0]+2*a,c-t_size[1]-a)
        cv2.rectangle(img, start_point, end_point, color, -1)   # Draw rectangle
        cv2.rectangle(img, start_point, end_point, (0,0,0), 2) # Draw border
        cv2.putText(img, text, (start_point[0]+int(a//2),start_point[1]-int(a//2)), cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale, (0, 0, 0), thick, lineType=cv2.LINE_AA) # Draw text

        return img
    
    # Prepare dataset for training
    def prepareData(self, path):
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        print("--------------------------------------------------")
        print("TRAIN")
        print("--------------------------------------------------")
        # Loop every folder in dataset
        folders = os.listdir(path)
        for folder in folders:
            if folder != '.DS_Store':
                TRAIN_PATH = f"{path}/{folder}/train"
                for cat_folder in os.listdir(TRAIN_PATH):
                    if cat_folder != '.DS_Store':
                        for img_path in os.listdir(f"{TRAIN_PATH}/{cat_folder}"):
                            if img_path != '.DS_Store':
                                # Read image
                                print(f"IMG : {TRAIN_PATH}/{cat_folder}/{img_path}")
                                img = cv2.imread(f"{TRAIN_PATH}/{cat_folder}/{img_path}",0)
                                # Extract feature
                                img_re = cv2.resize(img, self.img_size, interpolation = cv2.INTER_AREA)
                                descriptor = self.extractFeature(img_re)
                                x_train.append(descriptor)
                                # Label it
                                if cat_folder == "bottle":
                                    y_train.append(0)
                                elif cat_folder == "snack":
                                    y_train.append(1)
                                else:
                                    y_train.append(2)

        print("--------------------------------------------------")
        print("TEST")
        print("--------------------------------------------------")
        # Loop every folder in dataset
        folders = os.listdir(path)
        for folder in folders:
            if folder != '.DS_Store':
                TEST_PATH = f"{path}/{folder}/test"
                for cat_folder in os.listdir(TEST_PATH):
                    if cat_folder != '.DS_Store':
                        for img_path in os.listdir(f"{TEST_PATH}/{cat_folder}"):
                            if img_path != '.DS_Store':
                                # Read image
                                print(f"IMG : {TEST_PATH}/{cat_folder}/{img_path}")
                                img = cv2.imread(f"{TEST_PATH}/{cat_folder}/{img_path}",0)
                                # Extract feature
                                img_re = cv2.resize(img, self.img_size, interpolation = cv2.INTER_AREA)
                                descriptor = self.extractFeature(img_re)
                                x_test.append(descriptor)
                                # Label it
                                if cat_folder == "bottle":
                                    y_test.append(0)
                                elif cat_folder == "snack":
                                    y_test.append(1)
                                else:
                                    y_test.append(2)

        dataset = {
            "train": [np.array(x_train), np.array(y_train)],
            "test": [np.array(x_test), np.array(y_test)],
        }

        return dataset

    # Train with SVM
    def fit(self, dataset):

        clf = make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=42))
        train = dataset['train']
        clf.fit(train[0], train[1])

        return clf
    # Evaluate
    def result(sefl, model, dataset):
        test = dataset['test']
        grid_predictions = model.predict(test[0])

        print("")
        print(classification_report(test[1],grid_predictions))
        print("")
        print("Confustion martix")
        print(confusion_matrix(test[1], grid_predictions))

    # Prdict label
    def classify(self, fd):
        result = self.model.predict([fd])
        return result

    # Prdict and draw label on image
    def predict(self, img):
        img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fd = self.extractFeature(img_g)
        y_hat = self.classify(fd)
        text = ""
        if y_hat[0] == 0:
            text = "Bottle"
        elif y_hat[0] == 1:
            text = "Snack"
        elif y_hat[0] == 2:
            text = "Cans"
        img_draw = self.__drawText(img, text)
        return img_draw
    
    def saveModel(self, model, name="model.sav"):
        filename = name
        pickle.dump(model, open(filename, 'wb'))