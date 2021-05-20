# coding:utf-8
# 测试pypy


import run


@run.timethis
def testPypy():
    number = 0
    for i in range(100000000):
        number += i
        
    print("完成")
    
    
if __name__ == "__main__":
    testPypy()
    