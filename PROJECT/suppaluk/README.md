# Suppaluk

- python 3.8 up

## How to run
```
1) pip3 install pipenv
```

```
2) cd suppaluk
```

```
3) pipenv shell
```

```
4) pip install -r requirements.txt
```

```
5) uvicorn main:app --reload
```

## Train with new dataset
```
1) pip3 install pipenv
```

```
2) cd suppaluk
```

```
3) pipenv shell
```

```
4) pip install -r requirements.txt
```

```
5) add new data to folder dataset by use this pattern
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