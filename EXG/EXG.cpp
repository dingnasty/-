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
	Mat src, src_red, src_exg, src_exg1, src_exg_opti;

	src = cv::imread("D:/Y0227/1x/Y0104_TD_20180701090000_01.jpg");
	resize(src, src_red, Size(), 0.3, 0.3);
	imshow("input image", src_red);
	int rowNumber = src_red.rows;
	int colNumber = src_red.cols;
	cvtColor(src_red, src_exg, COLOR_BGR2GRAY);
	Mat src_gb, src_gb1;
	cvtColor(src_red, src_gb, COLOR_BGR2GRAY);
	cvtColor(src_red, src_gb1, COLOR_BGR2GRAY);
	//////////////////////EXG-EXR����///////////////////////////////////
	for (int i = 0; i < rowNumber; i++)
	{
		for (int j = 0; j < colNumber; j++)
		{
			src_exg.at<uchar>(i, j) = 2 * src_red.at<Vec3b>(i, j)[1] - src_red.at<Vec3b>(i, j)[2] -
				src_red.at<Vec3b>(i, j)[0]- 1.4* src_red.at<Vec3b>(i, j)[2]+ src_red.at<Vec3b>(i, j)[1];
		}
	}
	//imshow("EXG-EXR image", src_exg);

	for (int i = 0; i < rowNumber; i++)
	{
		for (int j = 0; j < colNumber; j++)
		{
			src_gb.at<uchar>(i, j) = 2 * src_red.at<Vec3b>(i, j)[1] - src_red.at<Vec3b>(i, j)[2] -
				src_red.at<Vec3b>(i, j)[0];
		}
	}
	imshow("EXG image", src_gb);

	double  Otsu = 0;
	double  Opti_Otsu = 0;
	Otsu = threshold(src_gb, src_exg1, Otsu, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	cout << "OpenCVthresh=" << Otsu << endl;
	Opti_Otsu = Otsu + 60;
	threshold(src_gb, src_exg_opti, Opti_Otsu, 255, CV_THRESH_BINARY);
	imshow("threshold image", src_exg_opti);

	ThresholdCut(src_exg_opti, src_exg_opti, 2000, 0, 0);
	imshow("ThresholdCut image", src_exg_opti);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));//��֤������
	morphologyEx(src_exg_opti, src_exg_opti, MORPH_OPEN, kernel);
	imshow("MORPH_OPEN image", src_exg_opti);

	///////////////////////G-R_otsu����/////////////////////////////////
	for (int i = 0; i < rowNumber; i++)
	{
		for (int j = 0; j < colNumber; j++)
		{
			if (src_red.at<Vec3b>(i, j)[1] > src_red.at<Vec3b>(i, j)[2])
			{
				src_gb1.at<uchar>(i, j) = src_red.at<Vec3b>(i, j)[1] - src_red.at<Vec3b>(i, j)[2];
			}
			else
			{
				src_gb1.at<uchar>(i, j) = 0;
			}

		}
	}
	imshow("GB image", src_gb1);
	float otsu_gb=0;
	threshold(src_gb1, src_gb1, otsu_gb, 255, CV_THRESH_OTSU);
	imshow("threshold_gb image", src_gb1);
	//Mat kernel1 = getStructuringElement(MORPH_RECT, Size(5, 5));//��֤������
	//morphologyEx(src_gb1, src_gb1, MORPH_CLOSE, kernel1);
	//imshow("MORPH_OPEN_gb image", src_gb1);
	ThresholdCut(src_gb1, src_gb1, 100, 1, 0);
	ThresholdCut(src_gb1, src_gb1, 500, 0, 0);
	imshow("thresholdCut image", src_gb1);



	int result = 0;
	for (int i = 0; i < rowNumber; i++)
	{
		uchar* data = src_exg_opti.ptr<uchar>(i);
		for (int j = 0; j < colNumber; j++)
		{
			if (data[j] == 0)
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