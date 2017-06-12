#include"kmeans.h"
kmeans::kmeans(int it, int alpha, int centriod_num) {
    iterator_time = it;
    alpha = alpha;
    centriod_num = centriod_num;
}
kmeans::kmeans(int k) {
    centriod_num = k;
}

vector<vector<double>> kmeans::Initial(vector<vector<double>>samples) {
    vector<vector<double>>centriod(centriod_num);
    if (samples.empty() || samples[0].empty())
    {
        cout << "Samples Error" << endl;
        return centriod;
    }

    for (int i = 0; i < centriod_num; i++)
        centriod[i].assign(samples[i].begin(), samples[i].end());
    return centriod;


}

vector<vector<double>> kmeans::Itera_Compute(vector<vector<double>>samples, vector<int>&label) {
    int sample_num = samples.size(), feature_num = samples[0].size();
    vector<vector<double>>centriods = Initial(samples);
    vector<vector<int>>temp(centriod_num);
    int iterator = 0;
    double err = 1;
    while (iterator < iterator_time && err>alpha)
    {
        for (int i = 0; i < sample_num; i++)
        {
            double mn = INT_MAX;
            int index;
            for (int j = 0; j < centriod_num; j++)
            {
                double d = distance(centriods[j], samples[i], feature_num);
                if (mn>d) {
                    mn = d;
                    index = j;
                }
            }
            temp[index].push_back(i);

        }

        err = 0;
        for (int k = 0; k < centriod_num; k++)
        {
            for (int i = 0; i < feature_num; i++)
            {
                double sum = 0;
                for (int j = 0; j < temp[k].size(); j++)
                {
                    sum += samples[temp[k][j]][i];
                }
                double t = sum / temp[k].size();
                err += abs(centriods[k][i] - t);
                centriods[k][i] = t;

            }
        }
        iterator++;
        cout << "time " << iterator << " Err " << err << endl;

    }

    for (int i = 0; i < centriod_num; i++)
    {
        for (int j = 0; j < temp[i].size(); j++)
            label[temp[i][j]] = i;
    }

    return centriods;
}
double kmeans::distance(vector<double>num1, vector<double>num2, int feature_num) {
    double sum = 0;
    for (int i = 0; i < feature_num; i++)
        sum += pow((num2[i] - num1[i]), 2);
    return sum;
}