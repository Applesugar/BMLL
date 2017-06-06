#pragma once
#include<vector>
#include<iostream>
#include<math.h>
using namespace std;

//y = arg max{P(Y = label[i]) * PAI{P(X = x[j] | Y = label[i])}}

class NaiveBayes
{
public:
    double lamuda = 1.0;  //默认采用拉普拉斯平滑
    vector<vector<int>> input;  //输入的特征应满足条件独立性假设，显然需要为离散变量
    vector<int> label;  //1,2,3,4……类标
    int sample_num = 10;
    int feature_num = 2;
    int label_num = 2;  //默认做二分类问题，可修改
public:
    NaiveBayes();
    ~NaiveBayes();
    int predict(vector<int>& x);
};

