#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

using namespace std;

#define READ(tag, val)   \
if ( buffer == tag":")   \
    in >> val;           \
else if ( buffer == tag )\
    in >> buffer >> val;


#define matrix(a, b) matrix[a * dimension + b]

typedef struct _city {
    int n;
    float x;
    float y;
} city;


typedef struct _tsp {
//    string name;
    int numberOfCities;
    float * distance;
} TSP;

int checkPathPossible(TSP * tsp, int * path) {

    int from;
    int to;
    for (int i = 0; i < tsp->numberOfCities - 1; ++i) {
        from = path[i];
        to = path[i + 1];

        if (from == -1) {
            clog << "Illegal city in position: " << i << "!"<< endl;
            return 0;
        }
        if (to == -1) {
            clog << "Illegal city in position: " << i + 1 << "!"<< endl;
            return 0;
        }
        if (tsp->distance[from * tsp->numberOfCities + to] <= 0.0f) {
            clog << "Path impossibile: " << from << " -> " << to << endl;
            return 0;
        }
    }
    return 1;
}

int calculatePath(TSP * tsp, int * path) {

    int from;
    int to;
    float distance = 0.0f;
    for (int i = 0; i < tsp->numberOfCities - 1; ++i) {
        from = path[i];
        to = path[i + 1];

        if ( from == -1 || to == -1 ) return -1;
        
        distance += tsp->distance[from * tsp->numberOfCities + to];
    }

    from = path[tsp->numberOfCities - 1];
    to = path[0];
    distance += tsp->distance[from * tsp->numberOfCities + to];

    return (int) distance;
}

float * readMatrixFromNodes(ifstream &in, string nodeCoordType, string edgeWeightType, int dimension) {

    if (nodeCoordType == "TWOD_COORDS" || nodeCoordType == "") {

        city * cities = (city *) malloc(dimension * sizeof(city));

        for (int i = 0; i < dimension; ++i) {
            in >> cities[i].n;
            in >> cities[i].x;
            in >> cities[i].y;
        } 

        float * matrix = (float *) malloc(dimension * dimension * sizeof(float));

        for (int i = 0; i < dimension; ++i){
            for (int j = 0; j < dimension; ++j) {

                float xd = cities[i].x - cities[j].x;
                float yd = cities[i].y - cities[j].y;

                if (edgeWeightType == "EUC_2D") {
                    matrix(i, j) = round( sqrt (xd * xd + yd * yd));
                } else if ( edgeWeightType == "MAN_2D") {
                    matrix(i, j) = round ( abs(xd) + abs(yd) );
                } else if ( edgeWeightType == "MAX_2D") {
                    matrix(i, j) = max ( round(abs(xd)), round(abs(yd)) );
                } else {
                    clog << "EDGE_WEIGHT_TYPE: " << edgeWeightType << " not supported!" << endl;
                    exit(-2);
                }
            }
        }

        free(cities);

        return matrix;

    } else {
        clog << "NODE_COORD_TYPE: " << nodeCoordType <<" not supported!" << endl;
        exit(-2);
    }

    return NULL;
}

float * readMatrixFromEdges(ifstream &in, string edgeWieghtFormat, int dimension) {

    float * matrix = (float *) malloc(dimension * dimension * sizeof(float));

    if (edgeWieghtFormat == "FULL_MATRIX") {
        for (int i = 0; i < dimension; ++i){
            for (int j = 0; j < dimension; ++j) {
                in >> matrix(i, j);
            }
        }
    } else if (edgeWieghtFormat == "UPPER_ROW") {
        
        //Reading upper triangular matrix without diagonal
        for (int i = 0; i < dimension; ++i){
            matrix(i, i) = 0.0f;
            for (int j = i + 1; j < dimension; ++j) {
                in >> matrix(i, j);
                matrix(j, i) = matrix(i, j);
            }
        }

    } else {
        clog << "EDGE_WEIGHT_FORMAT: " << edgeWieghtFormat << " not supported!" << endl;
        exit(-3);
    }
    return matrix;
}


TSP * getTPSFromFile(string filename) {

    string buffer = "";

    float * matrix = NULL;
    
    string name;
    string type;
    int dimension;
    string edgeWeightType;
    string edgeWieghtFormat;
    string nodeCoordType;
    string displayDataType;


    ifstream in(filename);
    if ( !in ) {
        clog << "Error while loading TSP file: " << filename << endl;
        exit(-2);
    }
    
    in >> buffer;

    while ( !in.eof() ) {

        READ("NAME", name)
        else READ("TYPE", type)
        else READ("DIMENSION", dimension)
        else READ("EDGE_WEIGHT_TYPE", edgeWeightType)
        else READ("EDGE_WEIGHT_FORMAT", edgeWieghtFormat)
        else READ("NODE_COORD_TYPE", nodeCoordType)
        else READ("DISPLAY_DATA_TYPE", displayDataType)
        else if ( buffer == "NODE_COORD_SECTION" ) {
            matrix = readMatrixFromNodes(in, nodeCoordType, edgeWeightType, dimension);
        } else if ( buffer == "EDGE_WEIGHT_SECTION") {
            matrix = readMatrixFromEdges(in, edgeWieghtFormat, dimension);
        }

        in >> buffer;
    }

    in.close();

    cout << "*****  " << name << "  *****" << endl;
    cout << "TYPE: " << type << endl;
    cout << "DIMENSION: " << dimension << endl;
    cout << "EDGE_WEIGHT_TYPE:" << edgeWeightType << endl;
    cout << "EDGE_WEIGHT_FORMAT: " << edgeWieghtFormat << endl;
    cout << "DISPLAY_DATA_TYPE: " << displayDataType << endl;
    cout << endl;

    TSP * tsp = (TSP *) malloc( sizeof(TSP) );
//    tsp->name += name;
    tsp->numberOfCities = dimension;
    tsp->distance = matrix;

    return tsp;
}
