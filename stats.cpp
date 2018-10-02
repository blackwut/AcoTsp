#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <algorithm>
#include <map>

#define UNIT_S  1000000.0
#define UNIT_MS 1000.0
#define UNIT_US 1
#define UNIT_OUTPUT UNIT_S
#define UNIT_SYMBOL "s" 

#define TIME(val) (val) / UNIT_OUTPUT

std::vector<std::string> names = {
    "bays29",
    "d198",
    "pcb442",
    "rat783",
    "pr1002",
    "pcb1173",
    "rl1889",
    "pr2392",
    "fl3795"
};

std::vector<uint32_t> threads = {
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128
};

struct Row {
    std::string name;
    int32_t    mapWorkers;
    int32_t    farmWorkers;
    uint32_t    maxEpoch;
    long        timeMS;
    long        timeUS;
    float       bestTourLength;
    std::string checkTour;
    std::string checkTourLength;
    uint32_t    nBlocks;           // AcoGPU only
    uint32_t    nWarpsPerBlock;    // AcoGPU only
};

struct Stat
{
    std::string name;
    uint32_t samples     = 0;
    uint32_t workers     = 0;
    double   sum         = 0.0;
    double   min         = __DBL_MAX__;
    double   avg         = 0.0;
    double   max         = __DBL_MIN__;
    double   speedup     = 0.0;
    double   scalability = 0.0;
    double   efficiency  = 0.0;
    // uint32_t nBlocks        = 0;    // AcoGPU only
    // uint32_t nWarpsPerBlock = 0;    // AcoGPU only
};

std::vector<uint32_t> getAllKeys(std::multimap<uint32_t, Row> & rows) {

    std::vector<uint32_t> keys;

    for (auto it = rows.begin(); it != rows.end(); it++) {
        if ( std::find(keys.begin(), keys.end(), it->first) == keys.end() ) {
           keys.push_back(it->first);
        }
    }
    return keys;
}

std::vector<Stat> getStats(std::multimap<uint32_t, Row> & rows, std::vector<uint32_t> keys) {

    std::vector<Stat> stats;

    for (std::string & name : names) {

        for (uint32_t & k : keys) {
            Stat s;
            s.name = name;
            s.workers = k;

            auto range = rows.equal_range(k);
            for (auto i = range.first; i != range.second; ++i) {
                Row & r = i->second;
                if (r.name == name && r.mapWorkers >= 0 && r.farmWorkers >= 0) {
                    const double time = r.timeUS;
                    s.samples += 1;
                    s.sum = s.sum + time;
                    if (s.min > time) s.min = time;
                    if (s.max < time) s.max = time;
                }
            }

            if (s.samples > 0) {
                s.sum = s.sum - s.min - s.max;
                s.avg = s.sum / (s.samples - 2);
                stats.push_back(s);
            } else {
                s.sum = 0.0;
                s.min = 0.0;
                s.avg = 0.0;
                s.max = 0.0;
            }
        }
    }

    return stats;
}

// TSeq / TPar(n)
void updateSpeedup(std::vector<Stat> & stats) {

    for (std::string & name : names) {

        double TSeq = 0.0;
        for (Stat & s : stats) {
            if (s.name == name && s.workers == 0) {
                TSeq = s.avg;
                break;
            }
        }

        for (Stat & s : stats) {
            if (s.name == name) {
                if (s.workers == 0) {
                    s.speedup = 0.0;
                } else {
                    s.speedup = TSeq / s.avg;
                }
            }
        }
    }
}

// TPar(1) / TPar(n)
void updateScalability(std::vector<Stat> & stats) {

    for (std::string & name : names) {

        double TP1 = 0.0;
        for (Stat & s : stats) {
            if (s.name == name && s.workers == 1) {
                TP1 = s.avg;
                break;
            }
        }

        for (Stat & s : stats) {
            if (s.name == name) {
                if (s.workers == 0) {
                    s.scalability = 1.0;
                } else {
                    s.scalability = TP1 / s.avg;
                }
            }
        }
    }
}

// Tseq / (TPar(n) * n)
void updateEfficiency(std::vector<Stat> & stats) {

    for (std::string & name : names) {

        double TSeq = 0.0;
        for (Stat & s : stats) {
            if (s.name == name && s.workers == 0) {
                TSeq = s.avg;
                break;
            }
        }

        for (Stat & s : stats) {
            if (s.name == name) {
                if (s.workers == 0) {
                    s.efficiency = 0.0;
                } else {
                    s.efficiency = TSeq / (s.avg * s.workers);
                }
            }
        }
    }
}

void loadFile(const std::string & filename, std::multimap<uint32_t, Row> & rows) {
    
    std::ifstream in(filename);
    if ( !in ) {
        std::clog << "Error while loading file: " << filename << std::endl;
        exit(-1);
    }

    std::string unused;
    float one;
    float two;
    while ( !in.eof() ) {
        Row r;

        in >> unused
        >> r.name
        >> r.mapWorkers
        >> r.farmWorkers
        >> r.maxEpoch
        >> r.timeMS
        >> r.timeUS
        >> r.bestTourLength
        >> r.checkTour
        >> r.checkTourLength
        >> one
        >> two;

        r.nBlocks = two;
        r.nWarpsPerBlock = one;

        if (r.timeUS == 0) {
            r.mapWorkers = r.farmWorkers = -1;
        }

        rows.insert({r.farmWorkers, r});
    }

    in.close();
}

int main(int argc, char * argv[]) {

    argc--;
    argv++;

    std::multimap<uint32_t, Row> rows;

    for (uint32_t i = 0; i < argc; ++i) {
        loadFile(argv[i], rows);
    }

    std::vector<uint32_t> keys = getAllKeys(rows);

    // for (Row r : rows) {
    //     std::cout << r.name  << "\t"
    //     << r.mapWorkers      << "\t"
    //     << r.farmWorkers     << "\t"
    //     << r.maxEpoch        << "\t"
    //     << r.timeMS          << "\t"
    //     << r.timeUS          << "\t"
    //     << r.bestTourLength  << "\t"
    //     << r.checkTour       << "\t"
    //     << r.checkTourLength << "\t"
    //     << std::endl;
    // }

    std::vector<Stat> stats = getStats(rows, keys);
    updateSpeedup(stats);
    updateScalability(stats);
    updateEfficiency(stats);

    std::cout
    << std::setw(27) << "Problem"               << " & "
    << std::setw( 7) << "\\#Thrs"               << " & "
    << std::setw(11) << "Avg (" UNIT_SYMBOL ")" << " & "
    << std::setw(11) << "SpeedUp"               << " & "
    << std::setw(13) << "Scalability"           << " & "
    << std::setw(12) << "Efficiency"            << " \\\\ \\hline" << std::endl;

    for (Stat s : stats) {

        const std::string name   = (s.workers == 0) ? "\\multirow{"+std::to_string(threads.size())+"}{*}{"+s.name+"}" : "";
        const double speedup     = (s.workers == 0) ? 0.0 : s.speedup;
        const double scalability = (s.workers == 0) ? 0.0 : s.scalability;
        const double efficiency  = (s.workers == 0) ? 0.0 : s.efficiency;

        std::cout << std::setw(27) << std::setprecision(4) << std::fixed
        << std::setw(27) << name             << " & "
        << std::setw( 7) << s.workers        << " & "
        << std::setw(11) << TIME(s.avg)      << " & "
        << std::setw(11) << speedup          << " & "
        << std::setw(13) << scalability      << " & "
        << std::setw(12) << efficiency       << " \\\\";
        if (s.workers == threads[threads.size() - 1]) std::cout << " \\hline";
        std::cout << std::endl;
    }

    return 0;
}
