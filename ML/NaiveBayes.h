#pragma once
#include<vector>
#include<iostream>
#include<math.h>
using namespace std;

//y = arg max{P(Y = label[i]) * PAI{P(X = x[j] | Y = label[i])}}

class NaiveBayes
{
public:
    double lamuda = 1.0;  //Ĭ�ϲ���������˹ƽ��
    vector<vector<int>> input;  //���������Ӧ�������������Լ��裬��Ȼ��ҪΪ��ɢ����
    vector<int> label;  //1,2,3,4�������
    int sample_num = 10;
    int feature_num = 2;
    int label_num = 2;  //Ĭ�������������⣬���޸�
public:
    NaiveBayes();
    ~NaiveBayes();
    int predict(vector<int>& x);
};

