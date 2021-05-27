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
path_img = "./datasets/grabage_real/train/bottle/bt02.jpg"
img = model.readImg(path_img)
img_pred = model.predict(img)
print(f"Ground Truth: {path_img.split('/')[4]}")
model.showImg(img_pred)

