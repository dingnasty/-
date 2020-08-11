#include <iostream>
#include <string>
#include <io.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void ThresholdCut(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	//CheckMode: 0代表去除黑区域，1代表去除白区域; NeihborMode：0代表4邻域，1代表8邻域;


	int RemoveCount = 0;       //记录除去的个数
	//记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
		cout << "Mode: 去除白色小区域. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] < 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}
	else
	{
		cout << "Mode: 去除黑色小区域. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] > 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}

	vector<Point2i> NeihborPos;  //记录邻域点位置
	NeihborPos.push_back(Point2i(-1, 0));
	NeihborPos.push_back(Point2i(1, 0));
	NeihborPos.push_back(Point2i(0, -1));
	NeihborPos.push_back(Point2i(0, 1));
	if (NeihborMode == 1)
	{
		cout << "Neighbor mode: 8邻域." << endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	else cout << "Neighbor mode: 4邻域." << endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//开始检测
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********开始该点处的检查**********
				vector<Point2i> GrowBuffer;                                      //堆栈，用于存储生长点
				GrowBuffer.push_back(Point2i(j, i));
				Pointlabel.at<uchar>(i, j) = 1;
				int CheckResult = 0;                                               //用于判断结果（是否超出大小），0为未超出，1为超出

				for (int z = 0; z < GrowBuffer.size(); z++)
				{

					for (int q = 0; q < NeihborCount; q++)                                      //检查四个邻域点
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX < Src.cols && CurrY >= 0 && CurrY < Src.rows)  //防止越界
						{
							if (Pointlabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //邻域点加入buffer
								Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //更新邻域点的检查标签，避免重复检查
							}
						}
					}

				}
				if (GrowBuffer.size() > AreaLimit) CheckResult = 2;                 //判断结果（是否超出限定的大小），1为未超出，2为超出
				else { CheckResult = 1;   RemoveCount++; }
				for (int z = 0; z < GrowBuffer.size(); z++)                         //更新Label记录
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
				}
				//********结束该点处的检查**********


			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//开始反转面积过小的区域
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iData = Src.ptr<uchar>(i);
		uchar* iDstData = Dst.ptr<uchar>(i);
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 2)
			{
				iDstData[j] = CheckMode;
			}
			else if (iLabel[j] == 3)
			{
				iDstData[j] = iData[j];
			}
		}
	}

	cout << RemoveCount << " objects removed." << endl;
}

int main()
{
	Mat Src, Src_Red, Src_GR, Src_Threshold;

	Src = cv::imread("D:/Y0227/1x/Y0104_TD_20180731080000_01.jpg");
	resize(Src, Src_Red, Size(), 0.3, 0.3);
	imshow("input image", Src_Red);
	int rowNumber = Src_Red.rows;
	int colNumber = Src_Red.cols;
	cvtColor(Src_Red, Src_GR, COLOR_BGR2GRAY);

	

	///////////////////////G-R_otsu方法/////////////////////////////////
	for (int i = 0; i < rowNumber; i++)
	{
		for (int j = 0; j < colNumber; j++)
		{
			if (Src_Red.at<Vec3b>(i, j)[1] > Src_Red.at<Vec3b>(i, j)[2])
			{
				Src_GR.at<uchar>(i, j) = Src_Red.at<Vec3b>(i, j)[1] - Src_Red.at<Vec3b>(i, j)[2];
			}
			else
			{
				Src_GR.at<uchar>(i, j) = 0;
			}

		}
	}

	imshow("Src_GR image", Src_GR);
	float otsu_gb = 0;
	threshold(Src_GR, Src_Threshold, otsu_gb, 255, CV_THRESH_OTSU);
	imshow("threshold_gr image", Src_Threshold);
	//Mat kernel1 = getStructuringElement(MORPH_RECT, Size(5, 5));//保证是奇数
	//morphologyEx(src_gb1, src_gb1, MORPH_CLOSE, kernel1);
	//imshow("MORPH_OPEN_gb image", src_gb1);
	ThresholdCut(Src_Threshold, Src_Threshold, 100, 0, 0);
	imshow("Dst image", Src_Threshold);



	int result = 0;
	for (int i = 0; i < rowNumber; i++)
	{
		uchar* data = Src_Threshold.ptr<uchar>(i);
		for (int j = 0; j < colNumber; j++)
		{
			if (data[j] == 255)
			{
				result++;
			}
		}
	}

	cout << result << endl;
	int totalNumber = rowNumber * colNumber;
	cout << totalNumber << endl;
	float area = double(result) * 100 / totalNumber;
	printf("覆盖率为：%.2f%%\n", area);

	waitKey(0);
	return 0;

}