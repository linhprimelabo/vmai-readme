# v-mai-reforming-fashion-retail

## Open-pose setup
Clone the origin open tf-pose project  
```shell
git clone https://www.github.com/ildoonet/tf-openpose
```
Install requirements  
```shell
cd tf-openpose
pip install -r requirements.txt
```

install swig  

```shell
apt install swig
```

```shell
cd tf_pose/pafprocess
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

```shell
cd ../..
python setup.py install
```

```shell
cd models/graph/cmu
bash download.sh
```

test  
```shell
cd ../../..
python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
```
##V-mai project
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
