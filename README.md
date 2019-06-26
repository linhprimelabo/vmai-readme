# v-mai-reforming-fashion-retail

## Open-pose setup
Clone the origin open tf-pose project  
'''shell
git clone https://www.github.com/ildoonet/tf-openpose
'''
Install requirements  
'''shell
cd tf-openpose
pip install -r requirements.txt
'''

install swig  

'''shell
apt install swig
'''

'''shell
cd tf_pose/pafprocess
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
'''

'''shell
cd ../..
python setup.py install
'''

