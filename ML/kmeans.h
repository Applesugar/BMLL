#pragma once
#include<iostream>
#include<vector>
#include<algorithm>
#include<math.h>
using namespace std;
class kmeans {
public:
    kmeans() {};
    ~kmeans() {};
    kmeans(int k);
    kmeans(int it, int alpha, int centeriod_num);
    vector<vector<double>> Initial(vector<vector<double>>samples);
    vector<vector<double>>Itera_Compute(vector<vector<double>>samples, vector<int>&label);
    double distance(vector<double>num1, vector<double>num2, int feature_num);

private:
    int iterator_time = 300;
    int alpha = 0.003;
    int centriod_num = 5;
};