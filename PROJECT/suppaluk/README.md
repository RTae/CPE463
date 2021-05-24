# Suppaluk

- python 3.8 up

## :gear: How to run
1) Install pipenv to crate a virtual environment
```
pip3 install pipenv
```

2) Change directory to suppaluk folder
```
cd suppaluk
```

3) create a virtual environment
```
pipenv shell
```

4) Install all package that need
```
pip install -r requirements.txt
```

5) Run fast api
```
uvicorn main:app --reload
```

6) go to
```
localhost:8000/docs
```

## :book: Train with new dataset
1) Install pipenv to crate a virtual environment
```
pip3 install pipenv
```

2) Change directory to suppaluk folder
```
cd suppaluk
```

3) create a virtual environment
```
pipenv shell
```

4) Install all package that need
```
pip install -r requirements.txt
```

5) add new data to folder dataset by use this pattern
```
    ./dataset
        --> new_datasetfolder
            --> train
                --> bottle
                    --> img1
                    --> img2
                    ...
                    --> imgN
                --> metal
                    --> img1
                    --> img2
                    ...
                    --> imgN
                --> snack
                    --> img1
                    --> img2
                    ...
                    --> imgN
            --> test
                --> bottle
                    --> img1
                    --> img2
                    ...
                    --> imgN
                --> metal
                    --> img1
                    --> img2
                    ...
                    --> imgN
                --> snack
                    --> img1
                    --> img2
                    ...
                    --> imgN
```

6) Edit model name and define path to image that test on new model in 'train.py'

7)  Run train.py for train model. It will show evaluation result and predict one image that you define the path
```
python train.py 
```
