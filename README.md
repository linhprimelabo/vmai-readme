# v-mai-reforming-fashion-retail

## 1. Open-pose setup
![openpose](https://github.com/ildoonet/tf-pose-estimation/blob/master/etcs/openpose_macbook_mobilenet3.gif?raw=true)
### Clone the origin open tf-pose project  
```shell
git clone https://www.github.com/ildoonet/tf-openpose
```
### Install requirements  
```shell
cd tf-openpose
pip install -r requirements.txt
```

### install swig  

```shell
apt install swig
```
cd to pafprocess folder and buid c++ lib for processing
```shell
cd tf_pose/pafprocess
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```
cd to tf-openpose folder and install package for later use.
```shell
cd ../..
python setup.py install
```
cd to cmu folder and download tensorflow graph file
```shell
cd models/graph/cmu
bash download.sh
```
### test
cd to tf-openpose folder and run the test demo
```shell
cd ../../..
python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
```
The result should be like this
![demo-pose](https://github.com/ildoonet/tf-pose-estimation/blob/master/etcs/inference_result2.png)
## 2. V-mai project
### Requirements
```
scikit-learn
tensorflow
keras
csv
request
shutil
pickle
lightgbm
xgboost
```
### Run test.py to check if the install succeed
```shell
cd ai-modules/src/codes
python test.py
```
if the install succeed, it should be show a result like this
![fin-result](test_result.png)
