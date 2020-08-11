#define DLLEXPORT extern "C" __declspec(dllexport)
#include"cpp2python.h"
#include <iostream>
#include <string.h>
#include <io.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#define DLLEXPORT extern "C" __declspec(dllexport)
using namespace cv;
using namespace std;


DLLEXPORT float c2p(const char* readPath,const char* savePath)
{
	Mat src, src_red, src_exg, src_exg1, src_exg_opti;

	src = cv::imread(readPath);
	resize(src, src_red, Size(), 0.3, 0.3);
	int rowNumber = src_red.rows;
	int colNumber = src_red.cols;
	cvtColor(src_red, src_exg, COLOR_BGR2GRAY);

	for (int i = 0; i < rowNumber; i++)
	{
		for (int j = 0; j < colNumber; j++)
		{
			src_exg.at<uchar>(i, j) = 2 * src_red.at<Vec3b>(i, j)[1] - src_red.at<Vec3b>(i, j)[2] -
				src_red.at<Vec3b>(i, j)[0] - 1.4 * src_red.at<Vec3b>(i, j)[2] + src_red.at<Vec3b>(i, j)[1];
		}
	}
	//imshow("EXG image", src_exg);
	double  Otsu = 0;
	double  Opti_Otsu = 0;
	Otsu = threshold(src_exg, src_exg1, Otsu, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	//cout << "OpenCVthresh=" << Otsu << endl;
	Opti_Otsu = Otsu - 20;
	threshold(src_exg, src_exg_opti, Opti_Otsu, 255, CV_THRESH_BINARY);
	//imshow("EXG image1", src_exg_opti);


	///////////////////�޳�С�������//////////////////////////////
		//CheckMode: 0����ȥ��������1����ȥ��������; NeihborMode��0����4����1����8����;


	int RemoveCount = 0;       //��¼��ȥ�ĸ���
	int CheckMode = 0, NeihborMode = 0, AreaLimit = 10;
	//��¼ÿ�����ص����״̬�ı�ǩ��0����δ��飬1�������ڼ��,2�����鲻�ϸ���Ҫ��ת��ɫ����3������ϸ������
	Mat Pointlabel = Mat::zeros(src_exg_opti.size(), CV_8UC1);
	Mat Dst = src_exg_opti.clone();

	if (CheckMode == 1)
	{
		//cout << "Mode: ȥ����ɫС����. ";
		for (int i = 0; i < src_exg_opti.rows; ++i)
		{
			uchar* iData = src_exg_opti.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < src_exg_opti.cols; ++j)
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
		//cout << "Mode: ȥ����ɫС����. ";
		for (int i = 0; i < src_exg_opti.rows; ++i)
		{
			uchar* iData = src_exg_opti.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < src_exg_opti.cols; ++j)
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
		//cout << "Neighbor mode: 8����." << endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	//else cout << "Neighbor mode: 4����." << endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//��ʼ���
	for (int i = 0; i < src_exg_opti.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < src_exg_opti.cols; ++j)
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
						if (CurrX >= 0 && CurrX < src_exg_opti.cols && CurrY >= 0 && CurrY < src_exg_opti.rows)  //��ֹԽ��
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
	for (int i = 0; i < src_exg_opti.rows; ++i)
	{
		uchar* iData = src_exg_opti.ptr<uchar>(i);
		uchar* iDstData = Dst.ptr<uchar>(i);
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < src_exg_opti.cols; ++j)
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

	//cout << RemoveCount << " objects removed." << endl;


	//////////////////////////////////���㸲�Ƕ�////////////////////////
	int result = 0;
	for (int i = 0; i < rowNumber; i++)
	{
		uchar* data = Dst.ptr<uchar>(i);
		for (int j = 0; j < colNumber; j++)
		{
			if (data[j] == 0)
			{
				result++;
			}
		}
	}

	//cout << result << endl;
	int totalNumber = rowNumber * colNumber;
	//cout << totalNumber << endl;
	float area = double(result) * 100 / totalNumber;
	imshow("EXG image2", Dst);
	imwrite(savePath, Dst);
	waitKey(0);
	//cout << area << endl;
	//printf("������Ϊ��%.2f%%\n", area);
	return area;

}