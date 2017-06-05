#pragma once
#include<iostream>
#include<vector>
#include<math.h>
using namespace std;
class LR
{
public:
    vector<double> res_w;
    double res_b;
public:
    double alpha = 0.1;
    int it_loops = 500;
    double e = 0.001;
    LR();
    ~LR();
    pair<vector<double>, double> buildLR(vector<vector<double>> & input, vector<int> & label, int sample_num, int feature_num);
    pair<vector<double>, double>LR::buildLRS(vector<vector<double>>&input, vector<int>& label, int sample_num, int feature_num);
    double sigmoid(vector<double> & w, vector<double> & x, double b, int feature_num);
    double predict(vector<double> &x, int feature_num);
};

