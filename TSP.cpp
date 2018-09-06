#ifndef __TSP_HPP__
#define __TSP_HPP__

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include "common.hpp"

using namespace std;

#define NAME "NAME"
#define TYPE "TYPE"
#define DIMENSION "DIMENSION"
#define TWOD_COORDS "TWOD_COORDS"
#define EDGE_WEIGHT_TYPE "EDGE_WEIGHT_TYPE"
#define EDGE_WEIGHT_FORMAT "EDGE_WEIGHT_FORMAT"
#define NODE_COORD_TYPE "NODE_COORD_TYPE"
#define DISPLAY_DATA_TYPE "DISPLAY_DATA_TYPE"
#define NODE_COORD_SECTION "NODE_COORD_SECTION"
#define EDGE_WEIGHT_SECTION "EDGE_WEIGHT_SECTION"
#define EUC_2D "EUC_2D"
#define MAN_2D "MAN_2D"
#define MAX_2D "MAX_2D"
#define FULL_MATRIX "FULL_MATRIX"
#define UPPER_ROW "UPPER_ROW"

#define READ(tag, val)   \
if ( buffer == tag":")   \
in >> val;           \
else if ( buffer == tag )\
in >> buffer >> val;

#define _edges(a, b) edges[a * dimension + b]

template <typename T>
struct City {
	int n;
	T x;
	T y;
};

template <typename T>
class TSP {
	
private:
	
	string name;
	string type;
	string edgeWeightType;
	string edgeWieghtFormat;
	string nodeCoordType;
	string displayDataType;

	void initEdges(const uint32_t size ) {
		edges = new T[size];
		for (uint32_t i = 0; i < size; ++i) {
			edges[i] = 0;
		}
	}
	
	void readEdgesFromNodes(ifstream &in) {
		
		if (TWOD_COORDS == nodeCoordType || nodeCoordType == "") {
			
			City<T> * cities = new City<T>[dimension];
			for (int i = 0; i < dimension; ++i) {
				in >> cities[i].n;
				in >> cities[i].x;
				in >> cities[i].y;
			}
			
			initEdges(dimension * dimension);
			
			for (int i = 0; i < dimension; ++i){
				for (int j = 0; j < dimension; ++j) {
					
					T xd = cities[i].x - cities[j].x;
					T yd = cities[i].y - cities[j].y;

					if ( edgeWeightType == EUC_2D ) {
						_edges(i, j) = round( sqrt(xd * xd + yd * yd) );
					} else if ( edgeWeightType == MAN_2D ) {
						_edges(i, j) = round( abs(xd) + abs(yd) );
					} else if ( edgeWeightType == MAX_2D ) {
						_edges(i, j) = max( round(abs(xd)), round(abs(yd)) );
					} else {
						clog << EDGE_WEIGHT_TYPE << ": " << edgeWeightType << " not supported!" << endl;
						exit(EXIT_EDGE_WEIGHT_TYPE);
					}
				}
			}
			
			delete[] cities;
			
		} else {
			clog << NODE_COORD_TYPE << ": " << nodeCoordType <<" not supported!" << endl;
			exit(EXIT_NODE_COORD_TYPE);
		}
	}
	
	void readEdgesFromEdges(ifstream &in) {
		
		initEdges(dimension * dimension);
		
		if ( edgeWieghtFormat == FULL_MATRIX ) {
			for (int i = 0; i < dimension; ++i){
				for (int j = 0; j < dimension; ++j) {
					in >> _edges(i, j);
				}
			}
		} else if ( edgeWieghtFormat == UPPER_ROW ) {
			
			//Reading upper triangular matrix without diagonal
			for (int i = 0; i < dimension; ++i){
				_edges(i, i) = 0.0f;
				for (int j = i + 1; j < dimension; ++j) {
					in >> _edges(i, j);
					_edges(j, i) = _edges(i, j);
				}
			}
			
		} else {
			clog << EDGE_WEIGHT_FORMAT << ": " << edgeWieghtFormat << " not supported!" << endl;
			exit(EXIT_EDGE_WEIGHT_FORMAT);
		}
	}
	
public:
	
	int dimension;
	T * edges;
	
	TSP(string filename) {
		string buffer = "";
		
		ifstream in(filename);
		if ( !in ) {
			clog << "Error while loading TSP file: " << filename << endl;
			exit(EXIT_LOAD_TSP_FILE);
		}
		
		in >> buffer;
		
		while ( !in.eof() ) {
			
			READ(NAME, name)
			else READ(TYPE, type)
			else READ(DIMENSION, dimension)
			else READ(EDGE_WEIGHT_TYPE, edgeWeightType)
			else READ(EDGE_WEIGHT_FORMAT, edgeWieghtFormat)
			else READ(NODE_COORD_TYPE, nodeCoordType)
			else READ(DISPLAY_DATA_TYPE, displayDataType)
			else if ( buffer == NODE_COORD_SECTION ) {
				readEdgesFromNodes(in);
			} else if ( buffer == EDGE_WEIGHT_SECTION) {
				readEdgesFromEdges(in);
			}
			
			in >> buffer;
		}
		
		in.close();
	}
	
	template <typename P>
	int checkPath(P * path) {
		
		uint32_t success = 1;
		uint32_t * duplicate = new uint32_t[dimension];

		for (uint32_t i = 0; i < dimension; ++i) {
			duplicate[i] = 0;
		}

		P from;
		P to;
		for (int i = 0; i < dimension - 1; ++i) {

			from = path[i];
			to = path[i + 1];

			duplicate[from] += 1;

			if (from < 0) {
				clog << "Illegal FROM city in position: " << i << "!"<< endl;
				success = 0;
			}
			if (to < 0) {
				clog << "Illegal TO city in position: " << i + 1 << "!"<< endl;
				success = 0;
			}
			if (_edges(from, to) <= 0) {
				clog << "Path impossibile: " << from << " -> " << to << endl;
				success = 0;
			}
		}

		for (uint32_t i = 0; i < dimension; ++i) {
			if (duplicate[i] > 1) {
				success = 0;
				clog << "Duplicate city: " << i << endl;
  			}
		}

		delete[] duplicate;

		return success;
	}
	
	template <typename P>
	T calculatePathLen(P * path) {
		
		T len = 0;
		for (uint32_t i = 0; i < dimension - 1; ++i) {
			const P from = path[i];
			const P to   = path[i + 1];
			
			len += _edges(from, to);
		}
		
		const P from = path[dimension - 1];
		const P to   = path[0];
		len += _edges(from, to);
		
		return len;
	}
	
	void printTSPInfo(TSP<T> * tsp) {
		cout << "*****  " << name << "  *****" << endl;
		cout << TYPE << ": " << type << endl;
		cout << DIMENSION << ": " << dimension << endl;
		cout << EDGE_WEIGHT_TYPE << ":" << edgeWeightType << endl;
		cout << EDGE_WEIGHT_FORMAT << ": " << edgeWieghtFormat << endl;
		cout << DISPLAY_DATA_TYPE << ": " << displayDataType << endl;
		cout << endl;
	}
	
	void printTSPEdges(TSP<T> * tsp) {
		printMatrix("Edges", edges, dimension, dimension);
	}
	
	string getName() {
		return name;
	}
	
	~TSP(){
		if (edges != NULL) delete[] edges;
	}
};

#endif

