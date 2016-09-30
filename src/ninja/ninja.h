#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <string.h>

#include <memory.h>
#include <time.h>
#include <ctime>

#include <iostream>

#include <sstream>

#include <iomanip>
#include <fstream>

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <sstream>
#include <cctype>
#include <cfloat>

#include "preconditioner.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef WINDNINJA_EXPORTS
#define WINDNINJA_API __declspec(dllexport)
#else
#define WINDNINJA_API
#endif

class WINDNINJA_API ninja
{
public:
    ninja(char *mat, char *col_indx, char *row_p, char *b);
    virtual ~ninja();
    
    bool UseSolver(int MAXITS, double stop_tol, int print_iters);
    void readSolution(char *filename);
    bool checkSolution();
//    ninja(const ninja &rhs);
//    ninja &operator=(const ninja &rhs);

   
    /*-----------------------------------------------------------------------------
     *
     *  Public Interface Methods
     *
     *
     *-----------------------------------------------------------------------------*/


protected:
 //   void checkCancel();//#include <string>
//    void write_compare_output();

private:

    double *DIAG;
    double *PHI, *RHS, *SK, *solution;
    int *row_ptr, *col_ind, NUMNP;
    double alphaH; //alpha horizontal from governing equation, weighting for change in horizontal winds
    double alpha;                //alpha = alphaH/alphaV, determined by stability

    bool solve(double *SK, double *RHS, double *PHI, int *row_ptr,
               int *col_ind, int NUMNP, int MAXITS, int print_iters, double stop_tol);
    // kk
    /*-----------------------------------------------------------------------------
     * alternative solvers                                                           
     *-----------------------------------------------------------------------------*/
    bool solveMinres(double *A, double *b, double *x, int *row_ptr, int *col_ind, int NUMNP, int max_iter, int print_iters, double tol);

    /*-----------------------------------------------------------------------------
     *  MKL Specific Functions
     *-----------------------------------------------------------------------------*/
    
    void cblas_dcopy(const int N, const double *X, const int incX,
                                        double *Y, const int incY);

    double cblas_ddot(const int N, const double *X, const int incX,
                                   const double *Y, const int incY);

    void cblas_daxpy(const int N, const double alpha, const double *X, const int incX,
                                                            double *Y, const int incY);

    double cblas_dnrm2(const int N, const double *X, const int incX);

    void mkl_dcsrmv(char *transa, int *m, int *k, double *alpha, char *matdescra,
                    double *val, int *indx, int *pntrb, int *pntre, double *x,
                    double *beta, double *y);

    void cblas_dscal(const int N, const double alpha, double *X, const int incX);
    void mkl_trans_dcsrmv(char *transa, int *m, int *k, double *alpha, char *matdescra, double *val, int *indx, int *pntrb, int *pntre, double *x, double *beta, double *y);

    /*-----------------------------------------------------------------------------
     *  End MKL Section
     *-----------------------------------------------------------------------------*/
    

    /* ----------------------------------------------------------------------------
     *  VDSpM  section
     *-----------------------------------------------------------------------------*/	
     void csr2vdspm (int *row_ptr, int *col_ind, double *data, int *numRows, int *elements, int *values,  int **start, int *numDiag, double ***diagonals);
     void vdspm_mv (int size, int *numRows, int *start, double **diagonals, double *p, double *w);
    /*-----------------------------------------------------------------------------
     *  End VDSpM Section
     *-----------------------------------------------------------------------------*/



    void interp_uvw();

    void write_A_and_b(int NUMNP, double *A, int *col_ind, int *row_ptr, double *b);
    double get_aspect_ratio(int NUMEL, int NUMNP, double *XORD, double *YORD, double *ZORD,
                            int nrows, int ncols, int nlayers);

    bool writePrjFile(std::string inPrjString, std::string outFileName);
    bool checkForNullRun();
    void discretize(); 
    void setBoundaryConditions();
    void computeUVWField();
    void prepareOutput();
    bool matched(int iter);
    void writeOutputFiles(); 
    void deleteDynamicMemory();
};
