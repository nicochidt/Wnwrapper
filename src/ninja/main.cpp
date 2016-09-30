#include <stdlib.h>

#include "ninja.h"

using namespace std;
int main(int argc, char** argv)
{
    if (argc < 5)
    {
        cout<<"Ussage " <<argv[0]<<" MatrixDataFile Row_ptrFile Col_ptrFile BFile Solution(optional)" <<endl;
        return -1;
    }
    bool compare = false;
    char *mat,  *col_indx,  *row_p,  *b;
    ninja simulation(argv[1], argv[3], argv[2], argv[4]);
    if (argc == 6)
    {   compare = true;
        simulation.readSolution(argv[5]);
    }
    
    
    int MAXITS = 100000;
    double stop_tol = 0.1;
    int print_iters = 10;
    if (!simulation.UseSolver(MAXITS, stop_tol, print_iters))
    {   cout<<"Solver failed\n"<<endl;
        return -1;
    }
    
    if (compare)
    {
        simulation.checkSolution();
    }
    
}