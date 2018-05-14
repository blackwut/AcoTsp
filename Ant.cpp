typedef struct _ant {
    int n;
    int next;
    int * visited;
    int * tabu;
    float * p;
    float length;
} Ant;

void clearAnt(Ant * ant) {
    
    ant->next = 0;

    for(int i = 0; i < ant->n; ++i) {
        ant->visited[i] = 0;
        ant->tabu[i] = 0;
        ant->p[i] = 0.0f;
   }

    ant->length = 0.0f;
}

Ant * newAnt(int n) {
    
    Ant * ant = (Ant *) malloc(sizeof(Ant));
    ant->n = n;
    ant->visited = (int *) malloc(ant->n * sizeof(int));
    ant->tabu = (int *) malloc(ant->n * sizeof(int));
    ant->p = (float *) malloc(ant->n * sizeof(float));
    
    clearAnt(ant);

    return ant;
}