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
	//CheckMode: 0����ȥ��������1����ȥ��������; NeihborMode��0����4����1����8����;


	int RemoveCount = 0;       //��¼��ȥ�ĸ���
	//��¼ÿ�����ص����״̬�ı�ǩ��0����δ��飬1�������ڼ��,2�����鲻�ϸ���Ҫ��ת��ɫ����3������ϸ������
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
		cout << "Mode: ȥ����ɫС����. ";
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
		cout << "Mode: ȥ����ɫС����. ";
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

	vector<Point2i> NeihborPos;  //��¼�����λ��
	NeihborPos.push_back(Point2i(-1, 0));
	NeihborPos.push_back(Point2i(1, 0));
	NeihborPos.push_back(Point2i(0, -1));
	NeihborPos.push_back(Point2i(0, 1));
	if (NeihborMode == 1)
	{
		cout << "Neighbor mode: 8����." << endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	else cout << "Neighbor mode: 4����." << endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//��ʼ���
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********��ʼ�õ㴦�ļ��**********
				vector<Point2i> GrowBuffer;                                      //��ջ�����ڴ洢������
				GrowBuffer.push_back(Point2i(j, i));
				Pointlabel.at<uchar>(i, j) = 1;
				int CheckResult = 0;                                               //�����жϽ�����Ƿ񳬳���С����0Ϊδ������1Ϊ����

				for (int z = 0; z < GrowBuffer.size(); z++)
				{

					for (int q = 0; q < NeihborCount; q++)                                      //����ĸ������
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX < Src.cols && CurrY >= 0 && CurrY < Src.rows)  //��ֹԽ��
						{
							if (Pointlabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //��������buffer
								Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //���������ļ���ǩ�������ظ����
							}
						}
					}

				}
				if (GrowBuffer.size() > AreaLimit) CheckResult = 2;                 //�жϽ�����Ƿ񳬳��޶��Ĵ�С����1Ϊδ������2Ϊ����
				else { CheckResult = 1;   RemoveCount++; }
				for (int z = 0; z < GrowBuffer.size(); z++)                         //����Label��¼
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
				}
				//********�����õ㴦�ļ��**********


			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//��ʼ��ת�����С������
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

	

	///////////////////////G-R_otsu����/////////////////////////////////
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
	//Mat kernel1 = getStructuringElement(MORPH_RECT, Size(5, 5));//��֤������
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
	printf("������Ϊ��%.2f%%\n", area);

	waitKey(0);
	return 0;

}