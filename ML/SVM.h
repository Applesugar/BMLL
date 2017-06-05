#pragma once
#include<iostream>
#include<vector>
#include<math.h>
using namespace std;

#define Linear 0
#define Polynormial 1
#define Gussian 2
#define Laplace 3

class SVM
{
public:
    vector<double> res_w;
    double res_b;
    vector<double> alpha;
    int KernelType = Linear;
    double Sigma = 0.1;   //高斯核和拉普拉斯核的分母
    int D = 2;    //多项式核的指数
public:
    vector<double> Error;
    double C = 0.1;
    int it_loops = 500;
    double e = 0.001;
    SVM();
    ~SVM();
    void initial(int sample_num, int feature_num);
    pair<int, int> out_loop(vector<vector<double>>& x, vector<int>& y, int sample_num, int feature_num);
    bool in_loop(int i, vector<vector<double>>& x, vector<int>& y, int sample_num, int feature_num);
    bool select_aj_1(int i, double Ei, vector<vector<double>>& x, vector<int>& y, int sample_num, int feature_num);
    bool select_aj_2(int i, vector<vector<double>>& x, vector<int>& y, int sample_num, int feature_num);
    bool select_aj_3(int i, vector<vector<double>>& x, vector<int>& y, int sample_num, int feature_num);
    bool optimize(int i, int j, vector<vector<double>>& x, vector<int>& y, int sample_num, int feature_num);
    pair<vector<double>, double> buildSVM(vector<vector<double>> &input, vector<int> & label, int sample_num, int feature_num);
//    int selectFisrt_alpha(vector<vector<double>> &input, vector<int> &label, int sample_num, int feature_num);
//    int selectSecond_alpha(vector<vector<double>> &input, vector<int> &label, int sample_num, int feature_num, int first);
    void calError(vector<vector<double>> &input, vector<int> &label, int sample_num, int feature_num);
    //KKT-conditions
    ///alpha[i] = 0  <=>  y[i] * g(x[i]) >= 1
    ///0 < alpha[i] < C  <=>  y[i] * g(x[i]) = 1
    ///alpha[i] = C  <=> y[i] * g(x[i]) <= 1
//    double hinge_lossFunction(vector<vector<double>>& x, vector<double>& y);
    double g(vector<vector<double>> &x, vector<int> &y, int sample_num, int feature_num, int i);
//    double inner_product(vector<double> &x1, vector<double> &x2, int feature_num);
    double inner_product(vector<double>& x1, vector<double>& x2, int feature_num);
    double PolynormialKernel(vector<double>& x1, vector<double>& x2, int feature_num, int d = 2);
    double GussianKernel(vector<double>& x1, vector<double>& x2, int feature_num, double sigma = 0.1);
    double LaplaceKernel(vector<double>& x1, vector<double>& x2, int feature_num, double sigma = 0.1);
    bool judgeToStop(vector<vector<double>> &x, vector<int> &y, int sample_num, int feature_num);
    int predict(vector<double> &x, int feature_num);
};

