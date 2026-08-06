# 尋找svm原始碼
# svm家族: SVM、SVC、SVR、LinearSVC、NuSVC...
import sklearn, inspect, os
from sklearn.svm import SVC

print(sklearn.__version__)                 # 版本
print(inspect.getsourcefile(SVC))          # SVC 的 Python 封裝檔路徑
import sklearn.svm as svm
print(os.path.dirname(svm.__file__))       # svm 套件目錄
print(os.listdir(os.path.dirname(svm.__file__)))
'''
1.7.2
D:\Users\Amanda\PycharmProjects\test\.venv1\Lib\site-packages\sklearn\svm\_classes.py
D:\Users\Amanda\PycharmProjects\test\.venv1\Lib\site-packages\sklearn\svm
['meson.build', 'src', 'tests', '_base.py', '_bounds.py', 
'_classes.py', '_liblinear.cp313-win_amd64.lib', 
'_liblinear.cp313-win_amd64.pyd', '_liblinear.pxi', 
'_liblinear.pyx', '_libsvm.cp313-win_amd64.lib', 
'_libsvm.cp313-win_amd64.pyd', '_libsvm.pxi',
'_libsvm.pyx', '_libsvm_sparse.cp313-win_amd64.lib',
'_libsvm_sparse.cp313-win_amd64.pyd', 
'_libsvm_sparse.pyx', '_newrand.cp313-win_amd64.lib', 
'_newrand.cp313-win_amd64.pyd', '_newrand.pyx', 
'__init__.py', '__pycache__']
'''

# ***主要四個檔案***
# _classes.py: 定義svc, svr, linearvc...
# _base.py: 和_classes共用邏輯
# _libsvm.pyx、_liblinear.pyx: 做訓練/預測的地方


