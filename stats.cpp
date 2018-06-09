#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

struct Row {
	string name;
	long nThreads;
	long ms;
	long ns;
	double length;
	string correct;
};

int nThreads[] = {1, 2, 4, 8, 16, 32, 64};

//rows = start pointer
//n = number of test with different nThread
//m = number of test with the same nThread
void printStatFor(Row * rows, int n, int m, ofstream &ff, ofstream &ideal) {
	
	double avgFirst = 0;
	for (int i = 0; i < n; ++i) {
		double sum = 0;
		double max = __DBL_MIN__;
		double min = __DBL_MAX__;
		double avg = 0;
		
		for (int j = 0; j < m; ++j) {
			double val = rows[i * m + j].ms;
			sum += val;
			if (val > max) max = val;
			if (val < min) min = val;
		}
		
		sum = sum - max - min;
		avg = sum / (m - 2);
		
		if ( i == 0 ) avgFirst = avg;
		
		ff << rows[i * m].nThreads << " " << avg << endl;
		ideal << nThreads[i] << " " <<  avgFirst / nThreads[i] << endl;
	}
}

int main(int argc, char * argv[]) {
	
//	int n = 0; //Number of rows
//	//int tsps = 9; //Number of tsp
//	int tests = 10; //Number of test with the same nThread
//	int nThread = 7; //Number of different nThreads nThreads[] =	{1, 2, 4, 8, 16, 32, 64};
	
	int n = 0;
	int tests = atoi(argv[2]);
	int nThread = atoi(argv[3]);
	
	
	ifstream in(argv[1]);
	if ( !in ) {
		clog << "Error while loading file: " << argv[1] << endl;
		exit(-1);
	}
	
	in >> n;
	Row rows[n];
	
	for (int i = 0; i < n; ++i) {
		in >> rows[i].name;
		in >> rows[i].nThreads;
		in >> rows[i].ms;
		in >> rows[i].ns;
		in >> rows[i].length;
		in >> rows[i].correct;;
	}
	
	in.close();
	
	for (int i = 0; i < n; i += tests * nThread) {
		ofstream ff("/Volumes/RamDisk/" + rows[i].name + "_ff.txt");
		ofstream ideal("/Volumes/RamDisk/" + rows[i].name + "_ideal.txt");

		printStatFor(rows + i, nThread, tests, ff, ideal);
		
		ff.close();
		ideal.close();
	}
	
//	for(int i = 0; i < n; ++i) {
//		cout << rows[i].name << " ";
//		cout << rows[i].nThreads << " ";
//		cout << rows[i].ms << " ";
//		cout << rows[i].ns << " ";
//		cout << rows[i].length << " ";
//		cout << rows[i].correct << " " << endl;
//	}
	
	return 0;
}
