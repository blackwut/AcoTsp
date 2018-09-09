#include <random>
#include <limits>
#include <cstddef>
#include <atomic>

#include <ff/parallel_for.hpp>
#include <ff/farm.hpp>

#include "Parameters.cpp"
#include "Environment.cpp"
#include "Ant.cpp"

using namespace ff;

template <typename T, typename D>
struct Emitter: ff_node_t< Ant<T> > {

    ff_loadbalancer * loadBalancer;
    const Parameters<T>     & params;
    const Environment<T, D> & env;
    std::vector< Ant<T> >   & ants;

    Emitter(ff_loadbalancer * loadBalancer,
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
        return (Ant<T> *)GO_ON;
    }
};

template <typename T, typename D>
struct Worker: ff_node_t< Ant<T> > {

    const Parameters<T> & params;
    Environment<T, D>   & env;

    std::mt19937 * generator = NULL;
    std::uniform_real_distribution<T> * distribution = NULL;

    Worker(const Parameters<T> & params, Environment<T, D> & env) :
    params(params),
    env   (env)
    {
        generator = new std::mt19937((unsigned int)time(0));
        distribution = new std::uniform_real_distribution<T>(0, 1);
    }

    const T nextRandom() {
        return distribution->operator()(*generator);
    }
    
    T atomic_addf(atomic<T> * f, T value){
        T old = f->load(std::memory_order_consume);
        T desired = old + value;
        while (!f->compare_exchange_weak(old, desired, std::memory_order_release, std::memory_order_consume)) {
            desired = old + value;
        }
        return desired;
    }
    
    Ant<T> * svc( Ant<T> * ant) {

        if (ant == (Ant<T> * )EOS) return (Ant<T> * )EOS;

        ant->constructTour(env.fitness, env.edges);
        const float tau = params.q / ant->getTourLength();

        auto bTabu = ant->getTabu().begin();
        const auto constbTabu = bTabu;

        while ( bTabu != ant->getTabu().end() - 1) {
            const uint32_t from = *(bTabu++);
            const uint32_t to   = *(bTabu);
            atomic_addf( &env.delta[from * env.nCities + to], tau );
        }
        const uint32_t from = *(bTabu);
        const uint32_t to   = *(constbTabu);
        atomic_addf( &env.delta[from * env.nCities + to], tau );

        return ant;
    }
    
    ~Worker() {
        delete generator;
        delete distribution;
    }
};

template <typename T, typename D>
class AcoFF {

private:
    const Parameters<T> & params;
    Environment<T, D>   & env;
    const uint32_t mapWorkers;
    const uint32_t farmWorkers;
    std::vector< Ant<T> > ants;

    ff_Farm<>            * farmTour;
    Emitter<T, D>        * emitterTour;
    ParallelForReduce<T> * pfr;

    void initEta(std::vector<T> & eta, 
                 const std::vector<T> & edges,
                 const uint32_t nCities)
    {
        const uint32_t elems = nCities * nCities;
        pfr->parallel_for(0L, elems, [&](const long i) {
            const T edgeVal = edges[i];
            eta[i] = (edgeVal == 0.0 ? 0.0 : 1.0 / edgeVal);
        });
    }

    void calcFitness(std::vector<T> & fitness,
                     const std::vector<T> & pheromone,
                     const std::vector<T> & eta,
                     const uint32_t nCities,
                     const T alpha,
                     const T beta)
    {
        const uint32_t elems = nCities * nCities;
        pfr->parallel_for(0L, elems, [&](const long i) {
            fitness[i] = pow(pheromone[i], alpha) * pow(eta[i], beta);
        });
    }

    void calcTour() {
        if (farmTour->run_then_freeze() < 0) {
            error("Running farm\n");
            exit(-1);
        }

        if (farmTour->wait_freezing() < 0) {
            error("Wait freezing farm\n");
            exit(-1);
        }
    }

    void updateBestTour(std::vector<uint32_t> & bestTour,
                        T & bestTourLength)
    {    
        // Ant<T> maxAnt(0);
        // Ant<T> * bestAnt;
        // pfr->parallel_reduce(bestAnt,
        //                      &maxAnt,
        //                      0L, env.nAnts,
        //                      [&](const long i, Ant<T> * minAnt) { 
        //                          minAnt = (ants[i] < *minAnt ? &ants[i] : minAnt); 
        //                      },
        //                      [](Ant<T> * minAnt, Ant<T> * ant) { 
        //                          minAnt = (*ant < *minAnt ? ant : minAnt);
        //                      });

        // std::copy ( bestAnt->getTabu().begin(), bestAnt->getTabu().end(), bestTour.begin() );
        // bestTourLength = bestAnt->getTourLength();
        const Ant<T> & bestAnt = *std::min_element(ants.begin(), ants.end());
        std::copy ( bestAnt.getTabu().begin(), bestAnt.getTabu().end(), bestTour.begin() );
        bestTourLength = bestAnt.getTourLength();
    }

    void resetDelta(std::vector<D> & delta,
                    const uint32_t nCities) {
        const uint32_t elems = nCities * nCities;
        pfr->parallel_for(0L, elems, [&](const long i) {
            delta[i] = 0.0;
        });
    }

    void updatePheromone(std::vector<T> & pheromone,
                         const std::vector<D> & delta,
                         const uint32_t nCities,
                         const T rho) {
        const uint32_t elems = nCities * nCities;
        pfr->parallel_for(0L, elems, [&](const long i) {
            pheromone[i] = pheromone[i] * rho + delta[i];
        });
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
    ants       (env.nAnts, Ant<T>(env.nCities))
    {
        farmTour = new ff_Farm<>( [&]() {
            std::vector< unique_ptr<ff_node> > workers;
            for(int i = 0; i < farmWorkers; ++i)
                workers.push_back( make_unique< Worker<T, D> >(params, env) );
            return workers;
        }());

        emitterTour = new Emitter<T, D>(farmTour->getlb(), params, env, ants);
        farmTour->add_emitter(*emitterTour);
        farmTour->remove_collector();
        farmTour->wrap_around();

        pfr = new ParallelForReduce<T>(mapWorkers);
        initEta(env.eta, env.edges, env.nCities);
    }

    void solve() {
        uint32_t epoch = 0;
        do {
            calcFitness    (env.fitness, env.pheromone, env.eta, env.nCities, params.alpha, params.beta);
            calcTour       ();
            updateBestTour (env.bestTour, env.bestTourLength);
            resetDelta     (env.delta, env.nCities);
            updatePheromone(env.pheromone, env.delta, env.nCities, params.rho);
        } while ( ++epoch < params.maxEpoch );
    }

    ~AcoFF(){
        delete farmTour;
        delete emitterTour;
        delete pfr;
    }
};
