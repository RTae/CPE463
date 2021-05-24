from src.model import suppaluk

# load model
model = suppaluk(train=True)

# prepare dataset
path = "./datasets"
dataset = model.prepareData(path)

# train
clf = model.fit(dataset)

# evaluate
model.result(clf, dataset)

# save model
name = "your_model_name"
model.saveModel(clf,name=f"{name}.sav")

# Test
model = suppaluk(path=f"{name}.sav")
img = model.readImg("./dataset/grabage_real/test/metal/c04.jpg")
img_pred = model.predict(img)
model.showImg(img_pred)

