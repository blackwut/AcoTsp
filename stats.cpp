#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <stdint.h>

std::vector<std::string> names = {
    "bays29",
    "d198",
    "pcb442",
    "rat783",
    "pr1002",
    "pcb1173",
    // "rl1889",
    // "pr2392",
    // "fl3795"
};

std::vector<uint32_t> threads = {
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64
};

struct Row {
    std::string name;
    uint32_t    mapWorkers;
    uint32_t    farmWorkers;
    uint32_t    maxEpoch;
    long        timeMS;
    long        timeUS;
    float       bestTourLength;
    std::string checkTour;
    std::string checkTourLength;
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
};

std::vector<Stat> getStats(std::vector<Row> & rows) {

    std::vector<Stat> stats;

    for (std::string & name : names) {

        for (uint32_t & thread : threads) {
            Stat s;
            s.name = name;
            s.workers = thread;

            for (Row & r : rows) {
                if (r.name == name && r.mapWorkers == thread && r.farmWorkers == thread) {
                    const double time = r.timeUS;
                    s.samples += 1;
                    s.sum = s.sum + time;
                    if (s.min > time) s.min = time;
                    if (s.max < time) s.max = time;

                }
            }

            s.sum = s.sum - s.min - s.max;
            s.avg = s.sum / (s.samples - 2);
            stats.push_back(s);
        }
    }

    return stats;
}

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

void loadFile(const std::string & filename, std::vector<Row> & rows) {
    
    std::ifstream in(filename);
    if ( !in ) {
        std::clog << "Error while loading file: " << filename << std::endl;
        exit(-1);
    }

    std::string unused;
    while ( !in.eof() ) {
    // for (uint32_t i = 0; i < 21; ++i) {

        Row r;

        in >> unused
        >> r.name
        >> r.mapWorkers
        >> r.farmWorkers
        >> r.maxEpoch
        >> r.timeMS
        >> r.timeUS
        >> r.bestTourLength
        // >> r.checkTour
        // >> r.checkTourLength
        // >> unused
        >> unused;

        rows.push_back(r);
    }

    in.close();
}

int main(int argc, char * argv[]) {

    argc--;
    argv++;

    std::vector<Row> rows;

    for (uint32_t i = 0; i < argc; ++i) {
        loadFile(argv[i], rows);
    }

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

    
    std::vector<Stat> stats = getStats(rows);
    updateSpeedup(stats);
    updateScalability(stats);
    updateEfficiency(stats);

    std::cout << std::setw(10)
        << std::setw(14) << "name"           << "\t"
        << std::setw(14) << "workers"        << "\t"
        << std::setw(14) << "samples"        << "\t"
        << std::setw(14) << "sum"            << "\t"
        << std::setw(14) << "min"            << "\t"
        << std::setw(14) << "avg"            << "\t"
        << std::setw(14) << "max"            << "\t"
        << std::setw(14) << "speedup"        << "\t"
        << std::setw(14) << "scalability"    << "\t"
        << std::setw(14) << "efficiency"     << "\t"
        << std::endl;

    for (Stat s : stats) {
        std::cout << std::setw(10) << std::setprecision(2) << std::fixed
        << std::setw(14) << s.name           << "\t"
        << std::setw(14) << s.workers        << "\t"
        << std::setw(14) << s.samples        << "\t"
        << std::setw(14) << s.sum            << "\t"
        << std::setw(14) << s.min            << "\t"
        << std::setw(14) << s.avg            << "\t"
        << std::setw(14) << s.max            << "\t"
        << std::setw(14) << s.speedup        << "\t"
        << std::setw(14) << s.scalability    << "\t"
        << std::setw(14) << s.efficiency     << "\t"
        << std::endl;
    }

    return 0;
}
