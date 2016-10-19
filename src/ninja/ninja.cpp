#include "ninja.h"

/**Ninja constructor
 * This is the default ninja constructor.
 */

ninja::ninja(char *mat, char *col_indx, char *row_p, char *b)
{
    std::ifstream Afd, col_indfd, row_ptrfd, bfd;
    int numnp, nznd, aux;
    Afd.open(mat, std::ios::in);
    col_indfd.open(col_indx, std::ios::in);
    row_ptrfd.open(row_p, std::ios::in);
    bfd.open(b, std::ios::in);
    
    std::cout << "Reading files"<<std::endl;
    if (Afd.is_open())
    {   Afd >> nznd;
		NZND = nznd;
#ifdef CUDA
		cudaMallocManaged(&SK, sizeof(double) * nznd);
#else
        SK = new double[nznd];
#endif
        for (int i = 0; i < nznd; i++)
            Afd >> SK[i];
    }
    
    else{
        std::cout << mat << " file"<<std::endl;
        throw std::runtime_error ("Error reading file");
    }
    if (bfd.is_open())
    {   bfd >> numnp;
        NUMNP = numnp;
        std::cout << b << " reading"<<std::endl;
#ifdef CUDA
		cudaMallocManaged(&RHS, sizeof(double) * numnp);
		cudaMallocManaged(&RHS, sizeof(double) * numnp);
		memset(RHS, 0, sizeof(double) * numnp);             
#else
        RHS = new double[numnp];
        PHI = new double[numnp]();   // initialized with zeros
#endif
        for (int i = 0; i < numnp; i++)
            bfd >> RHS[i];
        std::cout << b << " file readed"<<std::endl;
    }
    else{
        std::cout << mat << " file"<<std::endl;
        throw std::runtime_error ("Error reading file");
    }
    
    if (col_indfd.is_open())
    {   col_indfd >> aux;
        if (aux != nznd) throw std::runtime_error ("Incompatible files");
#ifdef CUDA
		cudaMallocManaged(&col_ind, sizeof(int) * nznd);
#else
        col_ind=new int[nznd];        //This holds the global column number of the corresponding element in the CRS storage
#endif
        for (int i = 0; i < nznd; i++)
            col_indfd >> col_ind[i];
        std::cout << col_indx << " file readed"<<std::endl;
    }
    else throw std::runtime_error ("Error reading file");
    
    if (row_ptrfd.is_open())
    {   row_ptrfd >> aux;
        aux -=1;
        if (aux != numnp) throw std::runtime_error ("Incompatible files");
#ifdef CUDA
		cudaMallocManaged(&row_ptr, sizeof(int) * (numnp + 1)); 
#else
        row_ptr=new int[numnp+1];     //This holds the element number in the SK array (CRS) of the first non-zero entry for the    
#endif
        for (int i = 0; i < numnp + 1; i++)
            row_ptrfd >> row_ptr[i];
        std::cout << row_ptrfd << " file readed"<<std::endl;
    }
    else throw std::runtime_error ("Error reading file");
    row_ptrfd.close();
    col_indfd.close();
    bfd.close();
    Afd.close();

}




/**Ninja destructor
 *
 */
ninja::~ninja()
{
	deleteDynamicMemory();
}

/*
 * Method to read the solution file
 * 
 * 
 */
void ninja::readSolution(char *filename)
{
        solution = new double[NUMNP];
        std::ifstream solutionfd;
        solutionfd.open(filename);
        if (solutionfd.is_open())
        {
            int num;
            solutionfd >> num;
            int it = 0;
            while(!solutionfd.eof())
            {
            //for (int it = 0; it < num; it++) 
                solutionfd >> solution[it];
                it++;
            }
            std::cout<<"Solution readed, "<<it-1<<" data"<<std::endl;
        }
        else throw std::runtime_error ("Error reading solution file");
}

/*
 * Method to check the correctness of the solution
 * 
 */
bool ninja::checkSolution()
{
    std::cout<<"Checking the solution"<<std::endl;
    double normb = cblas_dnrm2(NUMNP, RHS, 1);
    double *r = new double[NUMNP];
    double max = -999, t;
    
    for (int i = 0; i < NUMNP; i++)
    {
        t = PHI[i] - solution[i];
        r[i] = t;
        if (fabs(t)/PHI[i] > max) 
        {   std::cout<<PHI[i]<<" "<<solution[i]<<std::endl;
            max = fabs(t)/PHI[i];
        }
    }
    double resid = cblas_dnrm2(NUMNP, r, 1) / normb;
    std::cout<<"Module of the residue vector is "<<resid<<std::endl;
    std::cout<<"Max error: "<<max<<std::endl;
    
    
}


/*
 * Method to call the default solver
 * 
 * 
 * */
bool ninja::UseSolver(int MAXITS, double stop_tol, int print_iters)
{   std::cout << "About to solve" <<std::endl;
    bool value = solve(SK, RHS, PHI, row_ptr, col_ind, NUMNP, MAXITS, print_iters, stop_tol);
    std::cout << "Solved" <<std::endl;
    return value;
}





//  CG solver
//    This solver is fastest, but is not monotonic convergence (residual oscillates a bit up and down)
//    If this solver diverges, try MINRES from PetSc below...
/**Method called in ninja::simulate_wind() to solve the matrix equations.
 * This is a congugate gradient solver.
 * It seems to be the fastest, but is not monotonic convergence (residual 
oscillates a bit up and down).
 * If this solver diverges, try the MINRES from PetSc which is commented out 
below...
 * @param A Stiffness matrix in Ax=b matrix equation.  Storage is symmetric 
compressed sparse row storage.
 * @param b Right hand side of matrix equations.
 * @param x Vector to store solution in.
 * @param row_ptr Vector used to index to a row in A.
 * @param col_ind Vector storing the column number of corresponding value in A.
 * @param NUMNP Number of nodal points, so also the size of b, x, and row_ptr.
 * @param max_iter Maximum number of iterations to do.
 * @param print_iters How often to print out solver information.
 * @param tol Convergence tolerance to stop at.
 * @return Returns true if solver converges and completes properly.
 */
bool ninja::solve(double *A, double *b, double *x, int *row_ptr, int *col_ind, int NUMNP, int max_iter, int print_iters, double tol)
{
    
    //stuff for sparse BLAS MV multiplication
//    input.Com->ninjaCom(ninjaComClass::ninjaNone, "Inside the solver...");
    char transa='n';
    double one=1.E0, zero=0.E0;
    char matdescra[6];
    matdescra[0]='s';	//symmetric
    matdescra[1]='u';	//upper triangle stored
    matdescra[2]='n';	//non-unit diagonal
    matdescra[3]='c';	//c-style array (ie 0 is index of first element, not 1 like in Fortran)

    FILE *convergence_history;
    int i, j;
    double *p, *z, *q, *r;
    double alpha, beta, rho, rho_1, normb, resid;
    double residual_percent_complete, residual_percent_complete_old, time_percent_complete, start_resid;
    residual_percent_complete = 0.0;

    residual_percent_complete_old = -1.;

    Preconditioner M;
    if(M.initialize(NUMNP, A, row_ptr, col_ind, M.SSOR, matdescra)==false)
    {
  //      input.Com->ninjaCom(ninjaComClass::ninjaWarning, "Initialization of SSOR preconditioner failed, trying Jacobi preconditioner...");
        if(M.initialize(NUMNP, A, row_ptr, col_ind, M.Jacobi, matdescra)==false)
            throw std::runtime_error("Initialization of Jacobi preconditioner failed.");
    }

//#define NINJA_DEBUG_VERBOSE
#ifdef NINJA_DEBUG_VERBOSE
    if((convergence_history = fopen ("convergence_history.txt", "w")) == NULL)
        throw std::runtime_error("A convergence_history file to write to cannot be created.\nIt may be in use by another program.");

    fprintf(convergence_history,"\nIteration\tResidual\tResidual_check");
#endif //NINJA_DEBUG_VERBOSE

    p=new double[NUMNP];
    z=new double[NUMNP];
    q=new double[NUMNP];
    r=new double[NUMNP];
    

    //matrix vector multiplication A*x=Ax
    mkl_dcsrmv(&transa, &NUMNP, &NUMNP, &one, matdescra, A, col_ind, row_ptr, &row_ptr[1], x, &zero, r);

    for(i=0;i<NUMNP;i++){
        r[i]=b[i]-r[i];                  //calculate the initial residual
    }

    normb = cblas_dnrm2(NUMNP, b, 1);		//calculate the 2-norm of b
    //normb = nrm2(NUMNP, b);

    if (normb == 0.0)
        normb = 1.;

    //compute 2 norm of r
    resid = cblas_dnrm2(NUMNP, r, 1) / normb;
    //resid = nrm2(NUMNP, r) / normb;

    if (resid <= tol)
    {
        tol = resid;
        max_iter = 0;
        return true;
    }
#ifdef VDSpM
    
    double **diagonals;
    int *elements, *values, *start, numDiag;
    elements = new int[NUMNP];
    values = new int[NUMNP];
 
        
     
    csr2vdspm (row_ptr, col_ind, A, &NUMNP, elements, values, &start, &numDiag, &diagonals);
    
#endif
    
    //start iterating---------------------------------------------------------------------------------------
    for (int i = 1; i <= max_iter; i++)
    {

        M.solve(r, z, row_ptr, col_ind);	//apply preconditioner

        rho = cblas_ddot(NUMNP, z, 1, r, 1);
        //rho = dot(NUMNP, z, r);

        if (i == 1)
        {
            cblas_dcopy(NUMNP, z, 1, p, 1);
        }else {
            beta = rho / rho_1;

#pragma omp parallel for
            for(j=0; j<NUMNP; j++)
                p[j] = z[j] + beta*p[j];
        }

#ifdef VDSpM
	
	vdspm_mv (numDiag, &NUMNP, start, diagonals, p, q);
	
#else
        //matrix vector multiplication!!!		q = A*p;
        mkl_dcsrmv(&transa, &NUMNP, &NUMNP, &one, matdescra, A, col_ind, row_ptr, &row_ptr[1], p, &zero, q);
#endif
        alpha = rho / cblas_ddot(NUMNP, p, 1, q, 1);
        //alpha = rho / dot(NUMNP, p, q);

        cblas_daxpy(NUMNP, alpha, p, 1, x, 1);	//x = x + alpha * p;
        //axpy(NUMNP, alpha, p, x);

        cblas_daxpy(NUMNP, -alpha, q, 1, r, 1);	//r = r - alpha * q;
        //axpy(NUMNP, -alpha, q, r);

        resid = cblas_dnrm2(NUMNP, r, 1) / normb;	//compute resid
        //resid = nrm2(NUMNP, r) / normb;

        if(i==1)
            start_resid = resid;

        if((i%print_iters)==0)
        {

#ifdef NINJA_DEBUG_VERBOSE
    //        input.Com->ninjaCom(ninjaComClass::ninjaDebug, "Iteration = %d\tResidual = %lf\ttol = %lf", i, resid, tol);
    //        fprintf(convergence_history,"\n%ld\t%lf\t%lf",i,resid,tol);
#endif //NINJA_DEBUG_VERBOSE

            residual_percent_complete=100-100*((resid-tol)/(start_resid-tol));
            if(residual_percent_complete<residual_percent_complete_old)
                residual_percent_complete=residual_percent_complete_old;
            if(residual_percent_complete<0.)
                residual_percent_complete=0.;
            else if(residual_percent_complete>100.)
                residual_percent_complete=100.0;

            time_percent_complete=1.8*exp(0.0401*residual_percent_complete);
            if(time_percent_complete >= 99.0)
                time_percent_complete = 99.0;
            residual_percent_complete_old=residual_percent_complete;
            //fprintf(convergence_history,"\n%ld\t%lf\t%lf",i,residual_percent_complete, time_percent_complete);
      //      input.Com->ninjaCom(ninjaComClass::ninjaSolverProgress, "%d",(int) (time_percent_complete+0.5));
        }

        if (resid <= tol)	//check residual against tolerance
        {//   input.Com->ninjaCom(ninjaComClass::ninjaNone, "Residue %f",resid);
            break;
        }

        //cout<<"resid = "<<resid<<endl;

        rho_1 = rho;

    }	//end iterations--------------------------------------------------------------------------------------------

    if(p)
    {
        delete[] p;
        p=NULL;
    }
    if(z)
    {
        delete[] z;
        z=NULL;
    }
    if(q)
    {
        delete[] q;
        q=NULL;
    }
    if(r)
    {
        delete[] r;
        r=NULL;
    }

#ifdef NINJA_DEBUG_VERBOSE
//    fclose(convergence_history);
#endif //NINJA_DEBUG_VERBOSE
 //   input.Com->ninjaCom(ninjaComClass::ninjaNone, "Dumping solution...");
    std::ofstream solutionfd;
    solutionfd.open("SolutionB.txt");
    solutionfd << NUMNP << std::endl;
    
    for (int it = 0; it < NUMNP; it++) solutionfd << x[it] <<std::endl;
    solutionfd.close();
//        input.Com->ninjaCom(ninjaComClass::ninjaNone, "Done!...");

    if(resid>tol)
    {
        throw std::runtime_error("Solution did not converge.\nMAXITS reached.");
    }else{
        time_percent_complete = 100.0;
//        input.Com->ninjaCom(ninjaComClass::ninjaSolverProgress, "%d",(int) (time_percent_complete+0.5));
        return true;
    }
}

/**
 * @brief Computes the vector-matrix product A*x=y.
 *
 * This is a limited version of the BLAS function dcsrmv().
 * It is limited in that the matrix must be in compressed sparse row
 * format, and symmetric with only the upper triangular part stored.
 * ALPHA=1 and BETA=0 must also be true.
 *
 * @note My version of MKL's compressed sparse row (CSR) matrix vector product function
 * MINE ONLY WORKS FOR A SYMMETRICALLY STORED, UPPER TRIANGULAR MATRIX!!!!!!
 * AND ALPHA==1 AND BETA==0
 *
 * @param transa Not used here, but included to stay with the BLAS.
 * @param m Number of rows in "A" matrix. Must equal k in this implementation.
 * @param k Number of columns in "A" matrix. Must equal m in this implementation.
 * @param alpha Must be one for this implementation.
 * @param matdescra Not used here, but included to stay with the BLAS.
 * @param val This is the "A" array.  Must be in compressed sparse row format, and symmetric with only the upper triangular part stored.
 * @param indx An array describing the column index of "A" (sometimes called "col_ind").
 * @param pntrb A pointer containing indices of "A" of the starting locations of the rows.
 * @param pntre A pointer containg indices of "A" of the ending locations of the rows.
 * @param x Vector of size m (and k) in the A*x=y computation.
 * @param beta Not used here, but included to stay with the BLAS.
 * @param y Vector of size m (and k) in the A*x=y computation.
 */
void ninja::mkl_dcsrmv(char *transa, int *m, int *k, double *alpha, char *matdescra, double *val, int *indx, int *pntrb, int *pntre, double *x, double *beta, double *y)
{	// My version of MKL's compressed sparse row (CSR) matrix vector product function
	// MINE ONLY WORKS FOR A SYMMETRICALLY STORED, UPPER TRIANGULAR MATRIX!!!!!!
	// AND ALPHA==1 AND BETA==0

		//function multiplies a sparse matrix "val" times a vector "x", result is stored in "y"
		//		ie. Ax = y
		//"m" and "k" are equal to the number of rows and columns in "A" (must be equal in mine)
		//"indx" is an array describing the column index of "A" (sometimes called "col_ind")
		//"pntrb" is a pointer containing indices of "A" of the starting locations of the rows
		//"pntre" is a pointer containg indices of "A" of the ending locations of the rows
		int i,j,N;
		N=*m;

    #pragma omp parallel private(i,j)
    {
        #pragma omp for
        for(i=0;i<N;i++)
            y[i]=0.0;

        #pragma omp for
        for(i=0;i<N;i++)
        {
            y[i] += val[pntrb[i]]*x[i];	// diagonal
            for(j=pntrb[i]+1;j<pntre[i];j++)
            {
                y[i] += val[j]*x[indx[j]];
            }
        }
    }	//end parallel region

    for(i=0;i<N;i++)
    {
        for(j=pntrb[i]+1;j<pntre[i];j++)
        {
            {
                y[indx[j]] += val[j]*x[i];
            }
        }
    }
}

/**Computes the 2-norm of X.
 * A limited version of the BLAS function dnrm2().
 * @param N Size of X.
 * @param X Vector of size N.
 * @param incX Number of values to skip.  MUST BE 1 FOR THIS VERSION!!
 * @return Value of the 2-norm of X.
 */
double ninja::cblas_dnrm2(const int N, const double *X, const int incX)
{
	double val=0.0;
	int i;

	//#pragma omp parallel for reduction(+:val)
	for(i=0;i<N;i++)
		val += X[i]*X[i];
	val = std::sqrt(val);

	return val;
}

/**Performs the dot product X*Y.
 * A limited version of the BLAS function ddot().
 * @param N Size of vectors.
 * @param X Vector of size N.
 * @param incX Number of values to skip.  MUST BE 1 FOR THIS VERSION!!
 * @param Y Vector of size N.
 * @param incY Number of values to skip.  MUST BE 1 FOR THIS VERSION!!
 * @return Dot product X*Y value.
 */
double ninja::cblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY)
{
	double val=0.0;
	int i;

	#pragma omp parallel for reduction(+:val)
	for(i=0;i<N;i++)
		val += X[i]*Y[i];

	return val;
}

/**Copies values from the X vector to the Y vector.
 * A limited version of the BLAS function dcopy().
 * @param N Size of vectors.
 * @param X Source vector.
 * @param incX Number of values to skip.  MUST BE 1 FOR THIS VERSION!!
 * @param Y Target vector to copy values to.
 * @param incY Number of values to skip.  MUST BE 1 FOR THIS VERSION!!
 */
void ninja::cblas_dcopy(const int N, const double *X, const int incX, double *Y, const int incY)
{
	int i;
	for(i=0; i<N; i++)
		Y[i] = X[i];
}


/**
 * @brief Computes the vector-matrix product A*x=y using VDSpM representation for the matrix A.
 *
 * @note Only works if A is a symmetric matrix.
 *
 * @param size Number of diagonals with elements differents of 0
 * @param numRows Number of rows in "A" matrix. 
 * @param start Vector containing the position of the diagonal within the matrix A.
 * @param diagonals Matrix containing the elements of the matrix A. Each row represent a diagonal and have a different size.
 * @param p Vector of size numRows. Is the x vector in the A*x = y computation.
 * @param w Vector of size numRows. Is the y vector in the A*x = y computation.
 */
void ninja::vdspm_mv (int size, int *numrows, int *start, double **diagonals, double *p, double *w)
{
  int numRows = *numrows;
int j, i; 
int a=0;
int chunk= 5000;

#pragma omp parallel for default(shared) private(j)
for(j=0; j< numRows; j++)
	{
		w[j]=0.0;
	
	}
	#pragma omp parallel for schedule(static,chunk) private (j)
	for(j=0; j< numRows; j++)
	{
		w[j]+=diagonals[0][j]*p[j];
	}

	//#pragma omp parallel for private (i,j)
	for (i=1; i<size; i++)
	{
		a=start[i];
	#pragma omp parallel for default(shared) schedule(static,chunk) private (j)
		for (j=0; j< a; j++)
		{
			w[j]+=diagonals[i][j]*p[j+a];
		
		}
	}
	
		for (i=1; i<size; i++)
	{
		a=start[i];
	#pragma omp parallel for default(shared) schedule(static,chunk) private (j)
		for (j=a; j< numRows-a; j++)
		{
			w[j]+=diagonals[i][j]*p[j+a]+diagonals[i][j-a]*p[j-a];
		}
	}
	
			for (i=1; i<size; i++)
	{
		a=start[i];
	#pragma omp parallel for default(shared) schedule(static,chunk) private (j)
		for (j=numRows-a; j< numRows; j++)
		{
			w[j]+=diagonals[i][j-a]*p[j-a];
		}
	}
}



/**Performs the calculation Y = Y + alpha * X.
 * A limited version of the BLAS function daxpy().
 * @param N Size of vectors.
 * @param alpha
 * @param X Vector of size N.
 * @param incX Number of values to skip.  MUST BE 1 FOR THIS VERSION!!
 * @param Y Vector of size N.
 * @param incY Number of values to skip.  MUST BE 1 FOR THIS VERSION!!
 */
void ninja::cblas_daxpy(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY)
{
	int i;

	#pragma omp parallel for
	for(i=0; i<N; i++)
		Y[i] += alpha*X[i];
}


void cuda_alloc_csr_memory()
{
	double *cu_csr_row_ptr, double *cu_csr_col_ind, double *cu_csr_data, int data_size, int matrix_size
	cudaError_t custat1, custat2, custat3;
	custat1 = cudaMalloc((void**)&cu_csr_row_ptr, sizeof(int) * NUMNP)
}


void ninja::deleteDynamicMemory()
{
	if(col_ind)
	{	delete[] col_ind;
		col_ind=NULL;
	}
	if(row_ptr)
	{	delete[] row_ptr;
		row_ptr=NULL;
	}
	if(RHS)
	{	delete[] RHS;
		RHS=NULL;
	}
	if(PHI)
	{	delete[] PHI;
		PHI=NULL;
	}
	if(SK)
	{	delete[] SK;
		SK=NULL;
	}
	if(solution)
	{	delete[] solution;
		solution=NULL;
	}
}
