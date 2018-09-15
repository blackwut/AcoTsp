#include <random>
#include <limits>
#include <cstddef>
#include <atomic>
#include <thread>

#include <ff/parallel_for.hpp>
#include <ff/farm.hpp>

#include "Parameters.cpp"
#include "Environment.cpp"
#include "Ant.cpp"

using namespace ff;

template <typename T, typename D>
struct Emitter: ff_node_t< Ant<T> > {

    ff_loadbalancer * const loadBalancer;
    const Parameters<T>     & params;
    const Environment<T, D> & env;
    std::vector< Ant<T> >   & ants;

    Emitter(ff_loadbalancer * const loadBalancer,
            const Parameters<T>     & params,
            const Environment<T, D> & env,
            std::vector< Ant<T> >   & ants) :
    loadBalancer(loadBalancer),
    params(params),
    env   (env),
    ants  (ants)
    {}

    Ant<T> * svc(Ant<T> * s) {
        if (s == nullptr) {
            for (uint32_t i = 0; i < env.nAnts; ++i) {
                Ant<T> * ant = &ants[i];
                ff_node::ff_send_out(ant);
            }
        
            loadBalancer->broadcast_task(EOS);
        }
        return (Ant<T> *)GO_OUT;
    }
};

template <typename T, typename D>
struct Worker: ff_node_t< Ant<T> > {

    const Parameters<T> & params;
    Environment<T, D>   & env;

    Worker(const Parameters<T> & params, Environment<T, D> & env) :
    params(params),
    env   (env)
    {}
    
    inline T atomic_addf(std::atomic<T> * f, T value) {
        T old = f->load(std::memory_order_consume);
        T desired = old + value;
        while (!f->compare_exchange_weak(old, desired, 
                                         std::memory_order_release,
                                         std::memory_order_consume))
        {
            desired = old + value;
        }
        return desired;
    }
    
    Ant<T> * svc( Ant<T> * ant ) {

        if ((void *)ant == EOS) {
            return (Ant<T> * )EOS;
        }

        ant->constructTour(env.fitness, env.edges);
        const T tau = params.q / ant->getTourLength();

        auto bTabu = ant->getTabu().begin();
        const auto constbTabu = bTabu;

        while ( bTabu != ant->getTabu().end() - 1) {
            const uint32_t from = *(bTabu++);
            const uint32_t to   = *(bTabu);
            atomic_addf( &env.delta[from * env.nCities + to], tau );
            atomic_addf( &env.delta[to * env.nCities + from], tau );
        }
        const uint32_t from = *(bTabu);
        const uint32_t to   = *(constbTabu);
        atomic_addf( &env.delta[from * env.nCities + to], tau );
        atomic_addf( &env.delta[to * env.nCities + from], tau );

        return ant;
    }
    
    ~Worker() {}
};

template <typename T, typename D>
class AcoFF {

private:
    const Parameters<T> & params;
    Environment<T, D>   & env;
    const uint32_t mapWorkers;
    const uint32_t farmWorkers;
    std::vector< Ant<T> > ants;

    ParallelForReduce<T> pfr;
    ParallelForReduce< Ant<T> * > pfrAnts;

    void initEta(std::vector<T> & eta, 
                 const std::vector<T> & edges,
                 const uint32_t nCities)
    {
        const uint32_t elems = nCities * nCities;
        pfr.parallel_for(0L, elems, [&](const long i) {
            const T edgeVal = edges[i];
            eta[i] = (edgeVal == 0.0 ? 0.0 : 1.0 / edgeVal);
        });
        pfr.threadPause();
    }

    void calcFitness(std::vector<T> & fitness,
                     const std::vector<T> & pheromone,
                     const std::vector<T> & eta,
                     const uint32_t nCities,
                     const T alpha,
                     const T beta)
    {
        const uint32_t elems = nCities * nCities;
        pfr.parallel_for(0L, elems, [&](const long i) {
            fitness[i] = pow(pheromone[i], alpha) * pow(eta[i], beta);
        });
        pfr.threadPause();
    }

    void updateBestTour(std::vector<uint32_t> & bestTour,
                        T & bestTourLength)
    {
        Ant<T> maxAnt(0);
        Ant<T> * bestAnt = &ants[0];
        pfrAnts.parallel_reduce(bestAnt,
                                &maxAnt,
                                0L, env.nAnts - 1,
                                [&](const long i, Ant<T> * minAnt) {
                                   if (ants[i + 1] < *minAnt) minAnt = &ants[i + 1];
                                },
                                [](Ant<T> * minAnt, Ant<T> * ant) { 
                                   if (*ant < *minAnt) minAnt = ant;
                                });
        pfrAnts.threadPause();

        std::copy ( bestAnt->getTabu().begin(), bestAnt->getTabu().end(), bestTour.begin() );
        bestTourLength = bestAnt->getTourLength();
    }

    void resetDelta(std::vector<D> & delta,
                    const uint32_t nCities)
    {
        const uint32_t elems = nCities * nCities;
        pfr.parallel_for(0L, elems, [&](const long i) {
            delta[i] = 0.0;
        });
        pfr.threadPause();
    }

    void updatePheromone(std::vector<T> & pheromone,
                         const std::vector<D> & delta,
                         const uint32_t nCities,
                         const T rho) {
        const uint32_t elems = nCities * nCities;
        pfr.parallel_for(0L, elems, [&](const long i) {
            pheromone[i] = pheromone[i] * rho + delta[i];
        });
        pfr.threadPause();
    }

public:

    AcoFF(const Parameters<T> & params,
          Environment<T, D> & env,
          const uint32_t mapWorkers,
          const uint32_t farmWorkers) :
    params     (params),
    env        (env),
    mapWorkers (mapWorkers),
    farmWorkers(farmWorkers),
    ants       (env.nAnts, Ant<T>(env.nCities)),
    pfr        (mapWorkers),
    pfrAnts    (mapWorkers)
    {
        initEta(env.eta, env.edges, env.nCities);
    }

    void solve() {

        ff_Farm<> farmTour( [&]() {
            std::vector< std::unique_ptr< ff_node > > workers;
            for(uint32_t i = 0; i < farmWorkers; ++i)
                workers.push_back( make_unique< Worker<T, D> >(params, env) );
            return workers;
        } ());

        Emitter<T, D> E(farmTour.getlb(), params, env, ants);
        farmTour.add_emitter(E);
        farmTour.remove_collector();
        farmTour.wrap_around();

        uint32_t epoch = 0;
        do {
            calcFitness    (env.fitness, env.pheromone, env.eta, env.nCities, params.alpha, params.beta);

            // Calc Tour
            if (farmTour.run_then_freeze() < 0) { error("Running farm\n"); exit(EXIT_RUN_FARM); }
            if (farmTour.wait_freezing() < 0) { error("Wait freezing farm\n"); exit(EXIT_WAIT_FREEZING_FARM); }

            updateBestTour (env.bestTour, env.bestTourLength);
            resetDelta     (env.delta, env.nCities);
            updatePheromone(env.pheromone, env.delta, env.nCities, params.rho);
        } while ( ++epoch < params.maxEpoch );
    }

    ~AcoFF(){}
};
