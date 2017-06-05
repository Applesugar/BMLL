#include "LR.h"



LR::LR()
{
}


LR::~LR()
{
}

pair<vector<double>, double> LR::buildLR(vector<vector<double>> &input, vector<int> &label, int sample_num, int feature_num)
{
    vector<double> w(feature_num, 0.0);
    double b(0.0);
    double loss = 0.0;
    vector<double> delta_w(feature_num, 0.0);
    double delta_b = 0.0;
    double pre_loss = -1.0;
    for (int it_times = 0; it_times < it_loops; ++it_times) {
        if (abs(loss - pre_loss) <= e)
            break;
        pre_loss = loss;
        loss = 0;
        for (int i = 0; i < sample_num; ++i) {
            loss += label[i] * log(1 - sigmoid(w, input[i], b, feature_num)) + (1 - label[i]) * log(sigmoid(w, input[i], b, feature_num));
            for (int j = 0; j < feature_num; ++j) {
                delta_w[j] += input[i][j] * (label[i] - (1 - sigmoid(w, input[i], b, feature_num)));
            }
            delta_b += label[i] - (1 - sigmoid(w, input[i], b, feature_num));
        }
        loss /= -1.0 * sample_num;
        //¸üÐÂ w[] ºÍ b
        for (int i = 0; i < feature_num; ++i) {
            w[i] = w[i] - alpha * delta_w[i] / sample_num;
        }
        b = b - alpha * delta_b / sample_num;
    }
    

    res_w.assign(w.begin(), w.end());
    res_b = b;
    return pair<vector<double>, double>(w, b);
}
pair<vector<double>, double>LR::buildLRS(vector<vector<double>>&input, vector<int>& label, int sample_num, int feature_num)
{
    vector<double>w(feature_num, 0.0);
    double b = 0.0;
    double loss = 0;
    double pre_loss = -1;
    vector<double>delta_w(feature_num, 0.0);
    double delta_b = 0.0;
    for (int it_times = 0; it_times < it_loops; it_times++)
    {
        if (abs(loss - pre_loss) <= e)
            break;
        pre_loss = loss;
        loss = 0;
        for (int i = 0; i < sample_num; i++)
        {
            loss += label[i] * log(1 - sigmoid(w, input[i], b, feature_num)) + (1 - label[i])*(log(sigmoid(w, input[i], b, feature_num)));
        }
        loss = -loss / sample_num;
        int index = rand() % (sample_num - 1);
        cout << index << endl;
        for (int j = 0; j < feature_num; j++)
        {
            delta_w[j] = (label[index] - (1 - sigmoid(w, input[index], b, feature_num)))*input[index][j];
            w[j] = w[j] - alpha*delta_w[j];
        }
        delta_b = label[index] - (1 - sigmoid(w, input[index], b, feature_num));
        b = b - alpha*delta_b;
    }
    res_w.assign(w.begin(), w.end());
    res_b = b;
    return pair<vector<double>, double>(w, b);

}
double LR::sigmoid(vector<double>& w, vector<double>& x, double b, int feature_num)
{
    double wx = 0.0; // the inner product of w and x
    for (int i = 0; i < feature_num; ++i) {
        wx += w[i] * x[i];
    }
    double res = 1.0 / (1.0 + exp(-(wx + b)));
    return res;
}

double LR::predict(vector<double>& x, int feature_num)
{
    double P1 = 1 - sigmoid(res_w, x, res_b, feature_num);
    double P0 = 1 - P1;
    cout << "P(Y = 1 | x) = " << P1 << endl;
    cout << "P(Y = 0 | x) = " << P0 << endl;
    return P1 / P0;
}
