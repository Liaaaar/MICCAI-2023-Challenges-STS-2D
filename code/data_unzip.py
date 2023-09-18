import zipfile
#解压缩文件到路径
f = zipfile.ZipFile("traindata.zip",'r') # 压缩文件位置
for file in f.namelist():
    print(file)
    f.extract(file,"")               # 解压位置
f.close()
f = zipfile.ZipFile("testdata.zip",'r') # 压缩文件位置
for file in f.namelist():
    print(file)
    f.extract(file,"test/")               # 解压位置
f.close()
'''
文件目录
---train
    ---imgae
    ---mask
---test
    ---image
---infers
--models
---base.ipynb
'''