# encoding=UTF-8
"""
    @author: Administrator on 2017/6/18
    @email: ppsunrise99@gmail.com
    @step:
    @function: 
"""
from django.shortcuts import render


def hello(request):
    context = {}
    context['hello'] = 'Hello World!'
    return render(request, 'DUTNLP_WordSim.html', context)

if __name__ == '__main__':
    pass