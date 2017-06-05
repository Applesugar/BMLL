#include "SVM.h"
#include<math.h>
#include<time.h>
#include<algorithm>
using namespace std;


SVM::SVM()
{
}


SVM::~SVM()
{
}

void SVM::initial(int sample_num, int feature_num)
{
    res_w.assign(feature_num, 0.0);
    res_b = 0.0;
    alpha.assign(sample_num, 0.0);
    Error.assign(sample_num, 0.0);
}

pair<int, int> SVM::out_loop(vector<vector<double>>& x, vector<int>& y, int sample_num, int feature_num)
{
    int i, j;
    int changed = 0;
    bool flag = true; //是否需要遍历所有样本点？ false则只遍历支持向量
    if (flag) {
        for (int i = 0; i < sample_num; ++i) {
            changed += in_loop(i, x, y, sample_num, feature_num);
        }
    }
    else {
        for (int i = 0; i < sample_num; ++i) {
            if (alpha[i] > 0 && alpha[i] < C) {
                changed += in_loop(i, x, y, sample_num, feature_num);
            }

        }
    }
    if (flag)
        flag = false;
    if (!flag && changed == 0)
        flag = true;
    return pair<int, int>(i, j);
}

bool SVM::in_loop(int i, vector<vector<double>>& x, vector<int>& y, int sample_num, int feature_num)
{
    double yi = y[i];
    double ai = alpha[i];
    double Ei;
    if (Error[i] > 0 && Error[i] < C)
        Ei = Error[i];
    else
        Ei = g(x, y, sample_num, feature_num, i);
    double r = yi * Ei;
    //if(这个样本点在误差e的范围内违背了KKT条件的话） then 针对这个ai去选取aj
    if ((ai == 0 && r < 1 - e) || (ai > 0 && ai < C && abs(r - 1) > e) || (ai == C && r > 1 + e)) {
        if (select_aj_1(i, Ei, x, y, sample_num, feature_num))
            return 1;
        if (select_aj_2(i, x, y, sample_num, feature_num))
            return 1;
        if (select_aj_3(i, x, y, sample_num, feature_num))
            return 1;
    }
    return 0;
}
bool SVM::select_aj_1(int i, double Ei, vector<vector<double>>& x, vector<int>& y, int sample_num, int feature_num) {
    double max_E = 0.0;
    double E = 0.0;
    int index = -1;
    for (int j = 0; j < sample_num; ++j) {
        if (alpha[j] > 0 && alpha[j] < C)
            E = abs(Ei - Error[j]);
        if (E > max_E) {
            index = j;
            max_E = E;
        }
    }
    if (index >= 0 && optimize(i, index, x, y, sample_num, feature_num))
        return 1;
    else
        return 0;
}
bool SVM::select_aj_2(int i, vector<vector<double>>& x, vector<int>& y, int sample_num, int feature_num) {
    srand((unsigned)time(NULL));
    int start = rand() % sample_num;
    int index = 0;
    for (int j = 0; j < sample_num; ++j) {
        index = (j + start) % sample_num;
        if (alpha[index] > 0 && alpha[index] < C && optimize(i, index, x, y, sample_num, feature_num))
            return 1;
    }
    return 0;
}
bool SVM::select_aj_3(int i, vector<vector<double>>& x, vector<int>& y, int sample_num, int feature_num) {
    srand((unsigned)time(NULL));
    int start = rand() % sample_num;
    int index = 0;
    for (int j = 0; j < sample_num; ++j) {
        index = (j + start) % sample_num;
        if (optimize(i, index, x, y, sample_num, feature_num))
            return 1;
    }
    return 0;
}
bool SVM::optimize(int i, int j, vector<vector<double>>& input, vector<int>& label, int sample_num, int feature_num) {
    cout << "i = " << i << "   j = " << j << endl;
    if (i == j)
        return 0;
    calError(input, label, sample_num, feature_num);
    //求解新的alpha[i], alpha[j], 并更新
    double L, H;
    if (label[i] != label[j]) {
        L = max(0.0, alpha[j] - alpha[i]);
        H = min(C, C + alpha[j] - alpha[i]);
    }
    else {
        L = max(0.0, alpha[j] + alpha[i] - C);
        H = min(C, alpha[j] + alpha[i]);
    }
    if (L == H)
        return 0;
    double yita = inner_product(input[i], input[i], feature_num) + inner_product(input[j], input[j], feature_num) - 2 * inner_product(input[i], input[j], feature_num);
    double alpha_j_unc = alpha[j] + label[j] * (Error[i] - Error[j]) / yita;
    double alpha_j_old = alpha[j];  //更新之前先存一下，后面要用
    if (alpha_j_unc > H)
        alpha[j] = H;
    else if (alpha_j_unc < L)
        alpha[j] = L;
    else
        alpha[j] = alpha_j_unc;
    if (abs(alpha_j_old - alpha[j]) < e * (alpha_j_old + alpha[j] + e))
        return 0;
    double alpha_i_old = alpha[i];  //更新之前先存一下，后面要用
    alpha[i] = alpha_i_old + label[i] * label[j] * (alpha_j_old - alpha[j]);
    //更新res_w, res_b, Error

    double b_old = res_b;
    double b_new1 = -1 * Error[i] - label[i] * inner_product(input[i], input[i], feature_num) * (alpha[i] - alpha_i_old) - label[j] * inner_product(input[j], input[i], feature_num) * (alpha[j] - alpha_j_old) + b_old;
    double b_new2 = -1 * Error[j] - label[i] * inner_product(input[i], input[j], feature_num) * (alpha[i] - alpha_i_old) - label[j] * inner_product(input[j], input[j], feature_num) * (alpha[j] - alpha_j_old) + b_old;
    double b = 0.0;
    if (alpha[i] > 0 && alpha[i] < C)
        b = b_new1;
    else if (alpha[j] > 0 && alpha[j] < C)
        b = b_new2;
    else
        (b_new1 + b_new2) / 2.0;
    res_b = b;
    calError(input, label, sample_num, feature_num);
    vector<double> w(feature_num, 0.0);
    for (int i = 0; i < feature_num; ++i) {
        for (int j = 0; j < sample_num; ++j) {
            w[i] += alpha[j] * label[j] * input[j][i];
        }
    }
    res_w.assign(w.begin(), w.end());
    return 1;
}
pair<vector<double>, double> SVM::buildSVM(vector<vector<double>>& input, vector<int>& label, int sample_num, int feature_num)
{
    for (int it_times = 0; it_times < it_loops; ++it_times) {
        cout << "-----------------------第 " << it_times << " 次迭代中...----------------------" << endl;
        out_loop(input, label, sample_num, feature_num);
        if (judgeToStop(input, label, sample_num, feature_num))  //在e误差内停止迭代的条件
            break;
        //        int i = selectFisrt_alpha(input, label, sample_num, feature_num);
        //        int j = selectSecond_alpha(input, label, sample_num, feature_num, i);
    }
    return pair<vector<double>, double>(res_w, res_b);
}

//int SVM::selectFisrt_alpha(vector<vector<double>>& input, vector<int>& label, int sample_num, int feature_num)
//{
//    int index = 0;
//    double max_dis = -1.0;
//    for (int i = 0; i < sample_num; ++i) {
//        double kkt = label[i] * g(input, label, sample_num, feature_num, i);
//        double kkt_dis = 0.0;
//        /*if (alpha[i] == 0.0) {
//            if (kkt >= 1.0)
//                kkt_dis = 0.0;
//            else
//                kkt_dis = 1.0 - kkt;
//        }
//        else if (alpha[i] == C) {
//            if (kkt <= 1)
//                kkt_dis = 0;
//            else
//                kkt_dis = kkt - 1.0;
//        }*/
//        if(alpha[i] > 0 && alpha[i] < C) {   // 0 < alpha[i] < C
//            kkt_dis = abs(kkt - 1.0);
//        }
//        if (max_dis < kkt_dis) {
//            max_dis = kkt_dis;
//            index = i;
//        }
//    }
//    if (max_dis == 0.0) {
//        for (int i = 0; i < sample_num; ++i) {
//            double kkt = label[i] * g(input, label, sample_num, feature_num, i);
//            double kkt_dis = 0.0;
//            if (alpha[i] == 0.0) {
//                if (kkt >= 1.0)
//                    kkt_dis = 0.0;
//                else
//                    kkt_dis = 1.0 - kkt;
//            }
//            else if (alpha[i] == C) {
//                if (kkt <= 1)
//                    kkt_dis = 0;
//                else
//                    kkt_dis = kkt - 1.0;
//            }
//            if (max_dis < kkt_dis) {
//                max_dis = kkt_dis;
//                index = i;
//            }
//        }
//    }
//    return index;
//}

//int SVM::selectSecond_alpha(vector<vector<double>>& input, vector<int>& label, int sample_num, int feature_num, int first)
//{
//    //这个选择函数应该有些问题……
//    int index = 0;
//    calError(input, label, sample_num, feature_num);
//    double max_E = 0.0;
//    for (int j = 0; j < sample_num; ++j) {
//        if (alpha[j] > 0 && alpha[j] < C) {
//            double E = abs(Error[first] - Error[j]);
//            if (max_E < E) {
//                max_E = E;
//                index = j;
//            }
//        }
//    }
////    if (max_E <= e) {
//        srand((unsigned)time(NULL));
////        for(int i = 0;i < 10000000;++i){}
//        index = rand() % sample_num;
////    }
//    return index;
//}

void SVM::calError(vector<vector<double>>& input, vector<int>& label, int sample_num, int feature_num)
{
    //Error[i] = g(x[i]) - y[i]
    vector<double> error(sample_num, 0.0);
    for (int i = 0; i < sample_num; ++i) {
        for (int j = 0; j < sample_num; ++j) {
//            if(alpha[j] > 0 && alpha[j] < C)   //加了这个效果非常差
                error[i] += label[j] * alpha[j] * inner_product(input[i], input[j], feature_num);
        }
        error[i] = error[i] + res_b - label[i];
    }
    Error.assign(error.begin(), error.end());
}

double SVM::g(vector<vector<double>>& x, vector<int>& y, int sample_num, int feature_num, int i)
{
    double g = 0;
    for (int j = 0; j < sample_num; ++j) {
        g += alpha[j] * y[j] * inner_product(x[i], x[j],feature_num);
    }
    g += res_b;
    return g;
}

//double SVM::inner_product(vector<double>& x1, vector<double>& x2, int feature_num)
//{
//    double product = 0.0;
//    for (int i = 0; i < feature_num; ++i) {
//        product += x1[i] * x2[i];
//    }
//    return product;
//}
//重载内积计算函数，使其可以兼容其它核函数
double SVM::inner_product(vector<double>& x1, vector<double>& x2, int feature_num)
{
    int Kernel = KernelType;
    double sigma = Sigma;
    int d = D;
    switch (Kernel) {
    case Linear: {
        double product = 0.0;
        for (int i = 0; i < feature_num; ++i) {
            product += x1[i] * x2[i];
        }
        return product;
    }
    case Polynormial: {
        return PolynormialKernel(x1, x2, feature_num, d);
    }
    case Gussian: {
        return GussianKernel(x1, x2, feature_num, sigma);
    }
    case Laplace: {
        return LaplaceKernel(x1, x2, feature_num, sigma);
    }
    default: {
        return -1.0;
    }
    }
    
}

double SVM::PolynormialKernel(vector<double>& x1, vector<double>& x2, int feature_num, int d)
{
    double product = 0.0;
    for (int i = 0; i < feature_num; ++i) {
        product += x1[i] * x2[i];
    }
    product = pow(product, d);
    return product;
}

double SVM::GussianKernel(vector<double>& x1, vector<double>& x2, int feature_num, double sigma)
{
    double product = 0.0;
    for (int i = 0; i < feature_num; ++i) {
        product += pow(x1[i] - x2[i], 2);
    }
    product = -1 * product / (2 * pow(sigma, 2));
    product = exp(product);
    return product;
}

double SVM::LaplaceKernel(vector<double>& x1, vector<double>& x2, int feature_num, double sigma)
{
    double product = 0.0;
    for (int i = 0; i < feature_num; ++i) {
        product += pow(x1[i] - x2[i], 2);
    }
    product = -1 * sqrt(product) / sigma;
    product = exp(product);
    return product;
}

bool SVM::judgeToStop(vector<vector<double>> &x, vector<int> &y, int sample_num, int feature_num)
{
    double sum_ay = 0.0;
    for (int i = 0; i < sample_num; ++i) {
        if (alpha[i] < 0 || alpha[i] > C)
            return false;
        if (alpha[i] == 0) {
            if (y[i] * g(x, y, sample_num, feature_num, i) - 1.0 < -e)
                return false;
        }
        else if (alpha[i] == C) {
            if (y[i] * g(x, y, sample_num, feature_num, i) - 1.0 > e)
                return false;
        }
        else {
            if (abs(y[i] * g(x, y, sample_num, feature_num, i) - 1.0) > e)
                return false;
        }
//        sum_ay += alpha[i] * y[i];  //似乎不需要验证这个条件吧
    }
    return abs(sum_ay) <= e;
}

int SVM::predict(vector<double>& x, int feature_num)
{
    double f = inner_product(res_w, x, feature_num) + res_b;
    if (f > 0)
        return 1;
    else
        return -1;
}
