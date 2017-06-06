
#include<iostream>
#include<fstream>
#include<vector>
#include"LR.h"
#include"SVM.h"
#include"NaiveBayes.h"
using namespace std;

int main() {
    /*vector<vector<double>> x = {{0.5, 0.2}, {0.3, 0.1}, {0.2, 0.3}
                                ,{8, 6}, {7, 9},{8, 7}
                                };
    vector<int> y = { 1,1,1,0,0,0 };*/
    fstream A;
    A.open("D:\\ML\\BMLL\\2.txt",ios::in);
    if (!A.is_open()) {
        cout << "No Such Path or Training Text!" << endl;
        return -1;
    }
    vector<vector<int>> x(15, vector<int>(2, 0));
    vector<int> y(100, 0);
    int i = 0;
    while (!A.eof()) {
        A >> x[i][0];
        A >> x[i][1];
        A >> y[i];
        i++;
    }
    A.close();
    for (int i = 0; i < 15; ++i) {
        if (y[i] == -1)
            y[i] = 0;
        cout << x[i][0] << "  " << x[i][1] << "  " << y[i] << endl;
    }
    //testing the NaiveBayes
    NaiveBayes nb;
    nb.feature_num = 2;
    nb.input.assign(x.begin(), x.end());
    nb.label.assign(y.begin(), y.end());
    nb.label_num = 2;
    nb.sample_num = 15;
    nb.predict(vector<int>(2, 1));
    //testing the SVM
    //SVM svm;
    //svm.KernelType = Linear;
    //svm.Sigma = 2;
    //svm.D = 2;
    //svm.C = 1;
    //svm.e = 0.1;
    //svm.it_loops = 10;
    //svm.initial(100, 2);
    //svm.buildSVM(x, y, 70, 2);
    //int cnt_true = 0;
    //for (int i = 71; i < 100; ++i) {
    //    int res = svm.predict(x[i], 2);
    //    if (res == y[i])
    //        cnt_true++;
    //}
    //cout << "准确率为: " << cnt_true / 30.0 * 100 <<"%"<< endl;
    //cout << "w = ( " << svm.res_w[0] << " , " << svm.res_w[1] << " ), b = " << svm.res_b << endl;
    //testing the LR 
//    vector<double> test({ 1.3, 20 });
//    LR lr;
//    lr.alpha = 0.1;
//    lr.e = 0.00001;
//    lr.it_loops = 900000;
//    lr.buildLRS(x, y, 60, 2);
////    lr.predict(test, 2);
//    int count_1 = 0, count_0 = 0, count = 0;
//    for (int i = 61; i < 100; ++i) {
//        double pre = lr.predict(x[i], 2);
//        if (pre >= 1.0) {
//            count_1++;
//            if (y[i] == 1)
//                count++;
//        }
//        else {
//            count_0++;
//            if (y[i] == 0)
//                count++;
//        }
//    }
//    cout << "1类 = " << count_1 << endl << "0类 = " << count_0 << endl;
//    cout << "分类正确率：" << count / 40.0 * 100 << "%" << endl;
    return 0;
}