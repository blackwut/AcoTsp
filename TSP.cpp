#ifndef __TSP_CPP__
#define __TSP_CPP__

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <stdint.h>
#include <vector>

#include "common.hpp"

#define NAME                "NAME"
#define TYPE                "TYPE"
#define DIMENSION           "DIMENSION"
#define TWOD_COORDS         "TWOD_COORDS"
#define EDGE_WEIGHT_TYPE    "EDGE_WEIGHT_TYPE"
#define EDGE_WEIGHT_FORMAT  "EDGE_WEIGHT_FORMAT"
#define NODE_COORD_TYPE     "NODE_COORD_TYPE"
#define DISPLAY_DATA_TYPE   "DISPLAY_DATA_TYPE"
#define NODE_COORD_SECTION  "NODE_COORD_SECTION"
#define EDGE_WEIGHT_SECTION "EDGE_WEIGHT_SECTION"
#define EUC_2D              "EUC_2D"
#define MAN_2D              "MAN_2D"
#define MAX_2D              "MAX_2D"
#define ATT                 "ATT"
#define FULL_MATRIX         "FULL_MATRIX"
#define UPPER_ROW           "UPPER_ROW"

#define READ(tag, val)   \
if ( buffer == tag":")   \
in >> val;           \
else if ( buffer == tag )\
in >> buffer >> val;

#define _edges(a, b) edges[a * nCities + b]

template <typename T>
struct City {
    uint32_t n;
    T x;
    T y;
};

template <typename T>
class TSP {
    
private:
    
    uint32_t nCities;
    std::vector<T> edges;
    std::string name;
    std::string type;
    std::string edgeWeightType;
    std::string edgeWieghtFormat;
    std::string nodeCoordType;
    std::string displayDataType;

    void initEdges(const uint32_t size) {
        edges.resize(size, 0.0);
    }
    
    void readEdgesFromNodes(std::ifstream & in) {
        
        if (TWOD_COORDS == nodeCoordType || nodeCoordType == "") {
            
            std::vector< City<T> > cities(nCities);
            for (uint32_t i = 0; i < nCities; ++i) {
                in >> cities[i].n;
                in >> cities[i].x;
                in >> cities[i].y;
            }
            
            initEdges(nCities * nCities);
            
            for (uint32_t i = 0; i < nCities; ++i) {
                for (uint32_t j = 0; j < nCities; ++j) {
                    
                    const T xd = cities[i].x - cities[j].x;
                    const T yd = cities[i].y - cities[j].y;

                    if ( edgeWeightType == EUC_2D ) {
                        _edges(i, j) = round( sqrt(xd * xd + yd * yd) );
                    } else if ( edgeWeightType == MAN_2D ) {
                        _edges(i, j) = round( abs(xd) + abs(yd) );
                    } else if ( edgeWeightType == MAX_2D ) {
                        _edges(i, j) = std::max( round(abs(xd)), round(abs(yd)) );
                    } else if ( edgeWeightType == ATT ) {
                        const T rij = sqrt( (xd * xd + yd * yd) / 10.0 );
                        const T tij = round( rij );
                        _edges(i, j) = tij + (tij < rij);
                    } else {
                        std::cout << EDGE_WEIGHT_TYPE << ": " << edgeWeightType << " not supported!" << std::endl;
                        exit(EXIT_EDGE_WEIGHT_TYPE);
                    }
                }
            }

        } else {
            std::cout << NODE_COORD_TYPE << ": " << nodeCoordType <<" not supported!" << std::endl;
            exit(EXIT_NODE_COORD_TYPE);
        }
    }
    
    void readEdgesFromEdges(std::ifstream &in) {
        
        initEdges(nCities * nCities);
        
        if ( edgeWieghtFormat == FULL_MATRIX ) {
            for (uint32_t i = 0; i < nCities; ++i){
                for (uint32_t j = 0; j < nCities; ++j) {
                    in >> _edges(i, j);
                }
            }
        } else if ( edgeWieghtFormat == UPPER_ROW ) {
            
            //Reading upper triangular matrix without diagonal
            for (uint32_t i = 0; i < nCities; ++i){
                _edges(i, i) = 0.0f;
                for (uint32_t j = i + 1; j < nCities; ++j) {
                    in >> _edges(i, j);
                    _edges(j, i) = _edges(i, j);
                }
            }
            
        } else {
            std::cout << EDGE_WEIGHT_FORMAT << ": " << edgeWieghtFormat << " not supported!" << std::endl;
            exit(EXIT_EDGE_WEIGHT_FORMAT);
        }
    }
    
public:
    
    TSP(const std::string & filename) {
        std::string buffer = "";
        
        std::ifstream in(filename);
        if ( !in ) {
            std::cout << "Error while loading TSP file: " << filename << std::endl;
            exit(EXIT_LOAD_TSP_FILE);
        }
        
        in >> buffer;
        
        while ( !in.eof() ) {
            
            READ(NAME, name)
            else READ(TYPE,               type             )
            else READ(DIMENSION,          nCities          )
            else READ(EDGE_WEIGHT_TYPE,   edgeWeightType   )
            else READ(EDGE_WEIGHT_FORMAT, edgeWieghtFormat )
            else READ(NODE_COORD_TYPE,    nodeCoordType    )
            else READ(DISPLAY_DATA_TYPE,  displayDataType  )
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
    bool checkTour(const P path) const {
        
        bool success = true;
        std::vector<uint32_t> duplicate(nCities, 0);

        duplicate[path[0]] += 1;
        for (uint32_t i = 0; i < nCities - 1; ++i) {

            const uint32_t from = path[i];
            const uint32_t to   = path[i + 1];
            duplicate[to] += 1;

            if (from >= nCities) {
                std::cout << "Illegal FROM city in position: " << i << "!"<< std::endl;
                success = false;
            }
            if (success == true && to >= nCities) {
                std::cout << "Illegal TO city in position: " << i + 1 << "!"<< std::endl;
                success = false;
            }
            if (success == true && _edges(from, to) <= 0) {
                std::cout << "Path impossibile: " << from << " -> " << to << std::endl;
                success = false;
            }
        }

        for (uint32_t i = 0; i < nCities; ++i) {
            if (duplicate[i] > 1) {
                std::cout << "Duplicate city: " << i << std::endl;
                success = false;
            }
        }

        return success;
    }

    template <typename P>
    T calcTourLength(const P path) const {
        
        T totalLength = 0;
        for (uint32_t i = 0; i < nCities - 1; ++i) {
            const uint32_t from = path[i];
            const uint32_t to   = path[i + 1];
            
            totalLength += _edges(from, to);
        }
        
        const uint32_t from = path[nCities - 1];
        const uint32_t to   = path[0];
        totalLength += _edges(from, to);
        
        return totalLength;
    }
    
    void printInfo() const {
        std::cout << "*****  " << name << "  *****"       << std::endl
        << TYPE << ":               " << type             << std::endl
        << DIMENSION << ":          " << nCities          << std::endl
        << EDGE_WEIGHT_TYPE << ":   " << edgeWeightType   << std::endl
        << EDGE_WEIGHT_FORMAT << ": " << edgeWieghtFormat << std::endl
        << DISPLAY_DATA_TYPE << ":  " << displayDataType  << std::endl;
    }
    
    void printEdges() const {
        printMatrix("Edges", edges, nCities, nCities);
    }
    

    const std::vector<T> & getEdges() const {
        return edges;
    }

    uint32_t getNCities() const {
        return nCities;
    }

    const std::string & getName() const{
        return name;
    }
    
    ~TSP(){}
};

#endif
