#include "NaiveBayes.h"



NaiveBayes::NaiveBayes()
{
}


NaiveBayes::~NaiveBayes()
{
}

int NaiveBayes::predict(vector<int>& x)
{
    vector<double> P_label(label_num, lamuda);
    vector<double> cnt_label(label_num, 0.0);
    for (int i = 0; i < sample_num; ++i) {
        P_label[label[i]]++;
        cnt_label[label[i]]++;
    }
    for (int i = 0; i < label_num; ++i) {
        P_label[i] = P_label[i] / (double(label_num) + label_num * lamuda);
    }
    vector<vector<double>> cnt_feature(feature_num, vector<double>(label_num, lamuda));
    for (int i = 0; i < feature_num; ++i) {
        for (int j = 0; j < label_num; ++j) {
            for (int k = 0; k < sample_num; ++k) {
                if (input[k][i] == x[i] && label[k] == j)
                    cnt_feature[i][j]++;
            }
        }
    }
    for (int i = 0; i < feature_num; ++i) {
        for (int j = 0; j < label_num; ++j) {
            cnt_feature[i][j] = cnt_feature[i][j] / (cnt_label[j] + cnt_label[j] * lamuda);
        }
    }
    vector<double> P(P_label.begin(), P_label.end());
    for (int j = 0; j < label_num; ++j) {
        for (int i = 0; i < feature_num; ++i) {
            P[j] = P[j] * cnt_feature[i][j];
        }
    }
    double max = 0.0;

    int label_index = -1;
    for (int j = 0; j < label_num; ++j) {
        cout << "P(label = " << j << ") = " << P[j] << endl;
        if (max < P[j]) {
            max = P[j];
            label_index = j;
        }
    }
    cout << "该样本预测类别为：" << label_index << endl;
    return label_index;
}
