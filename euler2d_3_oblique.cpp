// oblique steady shock wave
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string.h>
#include <cmath>
#include <iomanip>
#include <math.h>
#include <vector>
#include <ctime>
#include <omp.h>

using namespace std;

double PI = acos(-1.0);
double CFL = 0.4;
const int m = 240;
const int n = 120;
vector<double> domain_x = {0, 6};
vector<double> domain_y = {0, 3};
double dx = (domain_x[1]-domain_x[0]) / double(m);
double dy = (domain_y[1]-domain_y[0]) / double(n);
double dt = pow(dx, 2.0);
double run_terminate = 20;
double gamma_val = 1.4;
double R_val = 287.05;
double Re_val = 1000.0;
double Pr_val = 0.73;

double initialization_rho(double x, double y);
double initialization_u(double x, double y);
double initialization_v(double x, double y);
double initialization_p(double x, double y);
double*** RungeKutta3(double dt, double*** U, double** T);
double TENO5(double a, double b, double c, double d, double e);
double WENO5(double a, double b, double c, double d, double e);
double CWENOZ_P(double a, double b, double c, double d, double e);
double CENO_P(double a, double b, double c, double d, double e, double *delta,double tre);
double MRWENO(double a, double b, double c, double d, double e);
double norm_Linf(double** f);
double diff_1(double a1, double a2, double a4, double a5, double h);
double diff_2(double a1, double a2, double a3, double a4, double a5, double b1, double b2, double b3, double b4, double b5, double h);
double* matmul(double** A, double* B);



double*** Tianpx = new double**[4];
double*** Tianmx = new double**[4];
double*** Tianpy = new double**[4];
double*** Tianmy = new double**[4];

double*** U = new double**[4];
double** T = new double*[m + 10];
double run_t = 0.0;
int main(int argc, const char** argv)
{
    
    for (int i = 0; i < 4; ++i) {
        Tianpx[i] = new double*[m + 10];
        Tianpy[i] = new double*[m + 10];
        Tianmx[i] = new double*[m + 10];
        Tianmy[i] = new double*[m + 10];
        for (int j = 0; j < m + 10; ++j) {
            Tianpx[i][j] = new double[n + 10];
            Tianpy[i][j] = new double[n + 10];
            Tianmx[i][j] = new double[n + 10];
            Tianmy[i][j] = new double[n + 10];
        }
    }
    for(int k = 0; k<4; k++){
        for(int i = 0; i < m+10; i++){
            for(int j = 0; j < n+10; j++){
                Tianpx[k][i][j] = 1111.0;
                Tianpy[k][i][j] = 1111.0;
                Tianmx[k][i][j] = 1111.0;
                Tianmy[k][i][j] = 1111.0;
            }
        }
    }
    
    
    double** func_rho = new double*[m + 10];
    double** func_u = new double*[m + 10];
    double** func_v = new double*[m + 10];
    double** func_p = new double*[m + 10];
    for (int i = 0; i < m + 10; ++i) {
        func_rho[i] = new double[n + 10];
        func_u[i] = new double[n + 10];
        func_v[i] = new double[n + 10];
        func_p[i] = new double[n + 10];
        T[i] = new double[n + 10];
    }
    for (int i = 0; i < 4; ++i) {
        U[i] = new double*[m + 10];
        for (int j = 0; j < m + 10; ++j) {
            U[i][j] = new double[n + 10];
        }
    }
    
    // initialization
    for(int i = 5; i < m+5; i++){
        for(int j = 5; j < n+5; j++){
            func_rho[i][j] = initialization_rho(domain_x[0]+(i-5)*dx, domain_y[0]+(j-5)*dy);
            func_u[i][j] = initialization_u(domain_x[0]+(i-5)*dx, domain_y[0]+(j-5)*dy);
            func_v[i][j] = initialization_v(domain_x[0]+(i-5)*dx, domain_y[0]+(j-5)*dy);
            func_p[i][j] = initialization_p(domain_x[0]+(i-5)*dx, domain_y[0]+(j-5)*dy);

            U[0][i][j] = func_rho[i][j];
            U[1][i][j] = func_rho[i][j] * func_u[i][j];
            U[2][i][j] = func_rho[i][j] * func_v[i][j];
            U[3][i][j] = func_p[i][j] / (gamma_val-1) + 0.5*func_rho[i][j]*(pow(func_u[i][j], 2) + pow(func_v[i][j], 2));
            T[i][j] = func_p[i][j] / (R_val * func_rho[i][j]);
        }
    }
    
    // time march
//   double start_time = omp_get_wtime();
    std::clock_t start = std::clock();
    while(run_t < run_terminate){
        
        double alphax = 0.0;
        double alphay = 0.0;
        for(int i = 5; i < m+5; i++){
            for(int j = 5; j < n+5; j++){
                alphax = max(alphax, fabs(U[1][i][j]/U[0][i][j]) + sqrt(gamma_val*(gamma_val-1)*(U[3][i][j]-0.5*(U[1][i][j]*U[1][i][j]+U[2][i][j]*U[2][i][j])/U[0][i][j])));
                alphay = max(alphay, fabs(U[2][i][j]/U[0][i][j]) + sqrt(gamma_val*(gamma_val-1)*(U[3][i][j]-0.5*(U[1][i][j]*U[1][i][j]+U[2][i][j]*U[2][i][j])/U[0][i][j])));
            }
        }
        dt = CFL / ( alphax/dx + alphay/dy);
//        dt = 1e-5;
        if ((run_terminate - run_t) <= dt){dt = run_terminate - run_t; }
        
      

        
        U = RungeKutta3(dt, U, T);
        for(int i = 5; i < m + 5; i++){
            for(int j = 5; j < n + 5; j++){
                T[i][j] = (gamma_val-1)*(U[3][i][j]-0.5*(U[1][i][j]*U[1][i][j]+U[2][i][j]*U[2][i][j]) / U[0][i][j]) / (R_val*U[0][i][j]);
            }
        }
        run_t = run_t + dt;
//        cout<<"run_t = "<< setprecision(12) << run_t <<endl;
        
    }
    std::clock_t end = std::clock();
    std::cout << "CPU Time taken: " << double(end - start) / CLOCKS_PER_SEC << " seconds\n";
//   double end_time = omp_get_wtime();
//   cout<<"Total time(s):"<<(double)(end_time-start_time)<<endl;
    
    // data saving
    for(int i = 5; i < m+5; i++){
        for(int j = 5; j < n+5; j++){
            func_rho[i][j] = U[0][i][j];
            func_u[i][j] = U[1][i][j]/U[0][i][j];
            func_v[i][j] = U[2][i][j]/U[0][i][j];
            func_p[i][j] = (gamma_val-1) * (U[3][i][j] - 0.5*func_rho[i][j]*(func_u[i][j]*func_u[i][j]+func_v[i][j]*func_v[i][j]));
        }
    }
    double error = norm_Linf(func_rho);
    cout<<"error_rho = "<< setprecision(4) <<  error <<endl;
    
    ofstream outFile_sol_rho;
    ofstream outFile_sol_u;
    ofstream outFile_sol_v;
    ofstream outFile_sol_p;
    outFile_sol_rho.open("sol_rho.txt");
    outFile_sol_u.open("sol_u.txt");
    outFile_sol_v.open("sol_v.txt");
    outFile_sol_p.open("sol_p.txt");
    for(int i = 5; i < m+5; i++){
        for(int j = 5; j < n+5; j++){
            outFile_sol_rho << setprecision(12) << func_rho[i][j]<<" ";
            outFile_sol_u << setprecision(12) << func_u[i][j]<<" ";
            outFile_sol_v << setprecision(12) << func_v[i][j]<<" ";
            outFile_sol_p << setprecision(12) << func_p[i][j]<<" ";
        }
        outFile_sol_rho<<endl;
        outFile_sol_u<<endl;
        outFile_sol_v<<endl;
        outFile_sol_p<<endl;
    }
    outFile_sol_rho.close();
    outFile_sol_u.close();
    outFile_sol_v.close();
    outFile_sol_p.close();
    
    for (int i = 0; i < m + 10; ++i) {
            delete[] func_rho[i];
            delete[] func_u[i];
            delete[] func_v[i];
            delete[] func_p[i];
            delete[] T[i];
        }
        delete[] func_rho;
        delete[] func_u;
        delete[] func_v;
        delete[] func_p;
        delete[] T;

    for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < m + 10; ++j) {
                delete[] U[i][j];
            }
            delete[] U[i];
        }
        delete[] U;
}


// time march function
double*** RungeKutta3(double dt, double*** U, double** T){

    double ***F = new double**[4];
    double ***G = new double**[4];
    double ***F_hat = new double**[4];
    double ***G_hat = new double**[4];
    double ***LU = new double**[4];
    double ***U_1 = new double**[4];
    double ***dF_nu_dx = new double**[4];
    double ***dG_nu_dy = new double**[4];
    for (int k = 0; k < 4; k++) {
        F[k] = new double*[m + 10];
        G[k] = new double*[m + 10];
        F_hat[k] = new double*[m + 10];
        G_hat[k] = new double*[m + 10];
        LU[k] = new double*[m + 10];
        U_1[k] = new double*[m + 10];
        dF_nu_dx[k] = new double*[m + 10];
        dG_nu_dy[k] = new double*[m + 10];
        for (int i = 0; i < m + 10; i++) {
            F[k][i] = new double[n + 10];
            G[k][i] = new double[n + 10];
            F_hat[k][i] = new double[n + 10];
            G_hat[k][i] = new double[n + 10];
            LU[k][i] = new double[n + 10];
            U_1[k][i] = new double[n + 10];
            dF_nu_dx[k][i] = new double[n + 10];
            dG_nu_dy[k][i] = new double[n + 10];
        }
    }

    for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < m+10; ++i) {
            for (int j = 0; j < n+10; ++j) {
                U_1[k][i][j] = U[k][i][j];
            }
        }
    }
    
    for(int rk = 0; rk < 3; rk++){
        
        // impose the boundary condition
        for(int i = 5; i < m + 5; i++){
            for(int ng = 0; ng < 5; ng++){
                if(domain_x[0]+(i-5)*dx<=0.5){
                    U[0][i][n+5+ng] = 1.0;
                    U[1][i][n+5+ng] = 4.0;
                    U[2][i][n+5+ng] = 0.0;
                    U[3][i][n+5+ng] = (1.0/1.4) / (gamma_val -1) + 0.5*16.0;
                }else{
                    U[0][i][n+5+ng] = 2.6667;
                    U[1][i][n+5+ng] = 2.6667*3.3750;
                    U[2][i][n+5+ng] = -2.6667*1.0825;
                    U[3][i][n+5+ng] = (3.2143) / (gamma_val -1) + 0.5*2.6667*(3.3750*3.3750 + 1.0825*1.0825);
                }
            }
        }
        for(int i = 5; i < m + 5; i++){
            for(int ng = 0; ng < 5; ng++){
                U[0][i][ng] = U[0][i][5];
                U[1][i][ng] = U[1][i][5];
                U[2][i][ng] = U[2][i][5];
                U[3][i][ng] = U[3][i][5];
            }
        }
        
        for(int j = 5; j < n + 5; j++){
            for(int ng = 0; ng < 5; ng++){
                U[0][ng][j] = 1.0;
                U[1][ng][j] = 4.0;
                U[2][ng][j] = 0.0;
                U[3][ng][j] = (1.0/1.4) / (gamma_val -1) + 0.5*16.0;
            }
        }
        for(int j = 5; j < n + 5; j++){
            for(int ng = 0; ng < 5; ng++){
                U[0][m+5+ng][j] = U[0][m+4][j];
                U[1][m+5+ng][j] = U[1][m+4][j];
                U[2][m+5+ng][j] = U[2][m+4][j];
                U[3][m+5+ng][j] = U[3][m+4][j];
            }
        }
        
        // redefine: F, G, H
#pragma omp parallel for collapse(2)
        for(int i = 0; i < m+10; i++){
            for(int j = 0; j < n+10; j++){
                double rho = U[0][i][j];
                double u = U[1][i][j]/U[0][i][j];
                double v = U[2][i][j]/U[0][i][j];
                double p = (gamma_val-1) * (U[3][i][j] - 0.5*rho*(u*u+v*v));
                
                F[0][i][j] = rho * u;
                F[1][i][j] = rho * u*u + p;
                F[2][i][j] = rho * u * v;
                F[3][i][j] = u * (U[3][i][j] + p);
                
                G[0][i][j] = rho * v;
                G[1][i][j] = rho * u * v;
                G[2][i][j] = rho * v*v + p;
                G[3][i][j] = v * (U[3][i][j] + p);
            }
        }
        
        // x direction
        // compute the maximum characteristic velocity
        double Lambda_x[4] = {0, 0, 0, 0};
        for(int i = 0; i < m+10; i++){
            for(int j = 0; j < n+10; j++){
                Lambda_x[0] = max(Lambda_x[0], fabs(fabs(U[1][i][j]/U[0][i][j]) - sqrt(gamma_val*(gamma_val-1)*(U[3][i][j] - 0.5*(U[1][i][j]*U[1][i][j] + U[2][i][j]*U[2][i][j])/U[0][i][j])) /U[0][i][j] ));
                Lambda_x[1] = max(Lambda_x[1], fabs(U[1][i][j]/U[0][i][j]));
                Lambda_x[2] = max(Lambda_x[3], fabs(fabs(U[1][i][j]/U[0][i][j]) + sqrt(gamma_val*(gamma_val-1)*(U[3][i][j] - 0.5*(U[1][i][j]*U[1][i][j] + U[2][i][j]*U[2][i][j])/U[0][i][j])) /U[0][i][j] ));
                Lambda_x[3] = Lambda_x[1];
            }
        }
        
#pragma omp parallel for collapse(2)
        for(int i = 4; i < m+6; i++){
            for(int j = 4; j < n+6; j++){
                
                double **U_pt = new double*[4];
                double **F_pt = new double*[4];
                double **L_x = new double*[4];
                double **R_x = new double*[4];
                double *U_mat = new double[4];
                double *F_mat = new double[4];
                double **u_pt = new double*[4];
                double **f_pt = new double*[4];
                double **f_p = new double*[4];
                double **f_m = new double*[4];
                double *f_hat = new double[4];
                for (int k = 0; k < 4; k++) {
                    U_pt[k] = new double[6];
                    F_pt[k] = new double[6];
                    L_x[k] = new double[4];
                    R_x[k] = new double[4];
                    u_pt[k] = new double[6];
                    f_pt[k] = new double[6];
                    f_p[k] = new double[5];
                    f_m[k] = new double[5];
                }

                // define stencil points
                for(int k = 0; k < 4; k++){
                    for(int pt = 0; pt < 6; pt++){
                        U_pt[k][pt] = U[k][i-2+pt][j];
                        F_pt[k][pt] = F[k][i-2+pt][j];
                    }
                }
                
                // projection
                double rho_temp1 = sqrt(U[0][i][j]) / (sqrt(U[0][i][j])+sqrt(U[0][i+1][j]));
                double rho_temp2 = sqrt(U[0][i+1][j]) / (sqrt(U[0][i][j])+sqrt(U[0][i+1][j]));
                double U0_temp = U[0][i][j] * rho_temp1 + U[0][i+1][j] * rho_temp2;
                double U1_temp = U[1][i][j] * rho_temp1 + U[1][i+1][j] * rho_temp2;
                double U2_temp = U[2][i][j] * rho_temp1 + U[2][i+1][j] * rho_temp2;
                double U3_temp = U[3][i][j] * rho_temp1 + U[3][i+1][j] * rho_temp2;
                double rho = U0_temp;
                double u = U1_temp / rho;
                double v = U2_temp / rho;
                double E = U3_temp;
                
                double p = (gamma_val-1) * (E-0.5*rho*(u*u+v*v));
                double H = (E + p) / rho;
                double c = sqrt(gamma_val * p / rho);
                double paraL = (gamma_val-1) / (2*c*c);

                double n_vector[2] = {1.0, 0.0};
                double l_vector[2] = {0.0, 1.0};
                double qn = u*n_vector[0]+v*n_vector[1];
                double ql = u*l_vector[0]+v*l_vector[1];
                L_x[0][0] = 0.5*((gamma_val-1)*(u*u+v*v)/(2*c*c) + qn/c);
                L_x[0][1] = -0.5*((gamma_val-1)*u/(c*c) + n_vector[0]/c);
                L_x[0][2] = -0.5*((gamma_val-1)*v/(c*c) + n_vector[1]/c);
                L_x[0][3] = paraL;
                L_x[1][0] = 1-paraL*(u*u+v*v);
                L_x[1][1] = 2*u * paraL;
                L_x[1][2] = 2*v * paraL;
                L_x[1][3] = -2 * paraL;
                L_x[2][0] = 0.5*((gamma_val-1)*(u*u+v*v)/(2*c*c) - qn/c);
                L_x[2][1] = -0.5*((gamma_val-1)*u/(c*c) - n_vector[0]/c);
                L_x[2][2] = -0.5*((gamma_val-1)*v/(c*c) - n_vector[1]/c);
                L_x[2][3] = paraL;
                L_x[3][0] = -ql;
                L_x[3][1] = l_vector[0];
                L_x[3][2] = l_vector[1];
                L_x[3][3] = 0.0;
                
                R_x[0][0] = 1.0;
                R_x[0][1] = 1.0;
                R_x[0][2] = 1.0;
                R_x[0][3] = 0.0;
                R_x[1][0] = u - c*n_vector[0];
                R_x[1][1] = u;
                R_x[1][2] = u+c*n_vector[0];
                R_x[1][3] = l_vector[0];
                R_x[2][0] = v-c*n_vector[1];
                R_x[2][1] = v;
                R_x[2][2] = v+c*n_vector[1];
                R_x[2][3] = l_vector[1];
                R_x[3][0] = H - qn*c;
                R_x[3][1] = 0.5*(u*u + v*v);
                R_x[3][2] = H + qn*c;
                R_x[3][3] = ql;
                
                for(int pt = 0; pt < 6; pt++){
                    U_mat[0] = U_pt[0][pt];
                    U_mat[1] = U_pt[1][pt];
                    U_mat[2] = U_pt[2][pt];
                    U_mat[3] = U_pt[3][pt];
                    double *u_mat = matmul(L_x, U_mat);
                    u_pt[0][pt] = u_mat[0];
                    u_pt[1][pt] = u_mat[1];
                    u_pt[2][pt] = u_mat[2];
                    u_pt[3][pt] = u_mat[3];

                    F_mat[0] = F_pt[0][pt];
                    F_mat[1] = F_pt[1][pt];
                    F_mat[2] = F_pt[2][pt];
                    F_mat[3] = F_pt[3][pt];
                    double *f_mat = matmul(L_x, F_mat);
                    f_pt[0][pt] = f_mat[0];
                    f_pt[1][pt] = f_mat[1];
                    f_pt[2][pt] = f_mat[2];
                    f_pt[3][pt] = f_mat[3];
                    
                    delete[] u_mat;
                    delete[] f_mat;
                }
                
                // flux splitting
                for(int k = 0; k < 4; k++){
                    for(int pt = 0; pt < 5; pt++){
                        f_p[k][pt] = 0.5 * (f_pt[k][pt] + Lambda_x[k]*u_pt[k][pt]);
                        f_m[k][pt] = 0.5 * (f_pt[k][5-pt] - Lambda_x[k]*u_pt[k][5-pt]);
                    }
                }
                
                // TENO reconstruction
                for(int k = 0; k < 4; k++){
                    
                    double AAAA=Tianpx[k][i][j];
                    double BBBB=Tianmx[k][i][j];
                    
//                    f_hat[k] = CWENOZ_P(f_p[k][0], f_p[k][1], f_p[k][2], f_p[k][3], f_p[k][4]) + CWENOZ_P(f_m[k][0], f_m[k][1], f_m[k][2], f_m[k][3], f_m[k][4]);
//                    f_hat[k] = TENO5(f_p[k][0], f_p[k][1], f_p[k][2], f_p[k][3], f_p[k][4]) + TENO5(f_m[k][0], f_m[k][1], f_m[k][2], f_m[k][3], f_m[k][4]);
                    
                    f_hat[k] = CENO_P(f_p[k][0], f_p[k][1], f_p[k][2], f_p[k][3], f_p[k][4], &AAAA, dx) + CENO_P(f_m[k][0], f_m[k][1], f_m[k][2], f_m[k][3], f_m[k][4], &BBBB, dx);
                    
//                    f_hat[k] = MRWENO(f_p[k][0], f_p[k][1], f_p[k][2], f_p[k][3], f_p[k][4]) + MRWENO(f_m[k][0], f_m[k][1], f_m[k][2], f_m[k][3], f_m[k][4]);
                    
                    Tianpx[k][i][j] = AAAA;
                    Tianmx[k][i][j] = BBBB;
                }
                
                // projection back
                double *F_hat_mat = matmul(R_x, f_hat);
                F_hat[0][i][j] = F_hat_mat[0];
                F_hat[1][i][j] = F_hat_mat[1];
                F_hat[2][i][j] = F_hat_mat[2];
                F_hat[3][i][j] = F_hat_mat[3];
                delete[] F_hat_mat;
                
                for (int k = 0; k < 4; k++) {
                    delete[] U_pt[k];
                    delete[] F_pt[k];
                    delete[] L_x[k];
                    delete[] R_x[k];
                    delete[] u_pt[k];
                    delete[] f_pt[k];
                    delete[] f_p[k];
                    delete[] f_m[k];
                }
                delete[] U_pt;
                delete[] F_pt;
                delete[] L_x;
                delete[] R_x;
                delete[] u_pt;
                delete[] f_pt;
                delete[] f_p;
                delete[] f_m;
                delete[] f_hat;
                delete[] U_mat;
                delete[] F_mat;
            }
        }

        
        // y direction
        // compute the maximum characteristic velocity
        double Lambda_y[4] = {0, 0, 0, 0};
        for(int i = 0; i < m+10; i++){
            for(int j = 0; j < n+10; j++){
                Lambda_y[0] = max(Lambda_y[0], fabs(fabs(U[2][i][j]/U[0][i][j]) - sqrt(gamma_val*(gamma_val-1)*(U[3][i][j] - 0.5*(U[1][i][j]*U[1][i][j] + U[2][i][j]*U[2][i][j])/U[0][i][j])) /U[0][i][j] ));
                Lambda_y[1] = max(Lambda_y[1], fabs(U[2][i][j]/U[0][i][j]));
                Lambda_y[2] =max(Lambda_y[3], fabs(fabs(U[2][i][j]/U[0][i][j]) + sqrt(gamma_val*(gamma_val-1)*(U[3][i][j] - 0.5*(U[1][i][j]*U[1][i][j] + U[2][i][j]*U[2][i][j])/U[0][i][j])) /U[0][i][j] ));
                Lambda_y[3] = Lambda_y[1];
            }
        }
        
#pragma omp parallel for collapse(2)
        for(int i = 4; i < m+6; i++){
            for(int j = 4; j < n+6; j++){

                double **L_y = new double*[4];
                double **R_y = new double*[4];
                double **U_pt = new double*[4];
                double **F_pt = new double*[4];
                double *U_mat = new double[4];
                double *F_mat = new double[4];
                double **u_pt = new double*[4];
                double **f_pt = new double*[4];
                double **f_p = new double*[4];
                double **f_m = new double*[4];
                double *f_hat = new double[4];
                for (int k = 0; k < 4; k++) {
                    L_y[k] = new double[4];
                    R_y[k] = new double[4];
                    U_pt[k] = new double[6];
                    F_pt[k] = new double[6];
                    u_pt[k] = new double[6];
                    f_pt[k] = new double[6];
                    f_p[k] = new double[5];
                    f_m[k] = new double[5];
                }
                // define stencil points
                for(int k = 0; k < 4; k++){
                    for(int pt = 0; pt < 6; pt++){
                        U_pt[k][pt] = U[k][i][j-2+pt];
                        F_pt[k][pt] = G[k][i][j-2+pt];
                    }
                }
                
                // projection
                double rho_temp1 = sqrt(U[0][i][j]) / (sqrt(U[0][i][j])+sqrt(U[0][i][j+1]));
                double rho_temp2 = sqrt(U[0][i][j+1]) / (sqrt(U[0][i][j])+sqrt(U[0][i][j+1]));
                double U0_temp = U[0][i][j] * rho_temp1 + U[0][i][j+1] * rho_temp2;
                double U1_temp = U[1][i][j] * rho_temp1 + U[1][i][j+1] * rho_temp2;
                double U2_temp = U[2][i][j] * rho_temp1 + U[2][i][j+1] * rho_temp2;
                double U3_temp = U[3][i][j] * rho_temp1 + U[3][i][j+1] * rho_temp2;
                double rho = U0_temp;
                double u = U1_temp / rho;
                double v = U2_temp / rho;
                double E = U3_temp;
                
                double p = (gamma_val-1) * (E-0.5*rho*(u*u+v*v));
                double H = (E + p) / rho;
                double c = sqrt(gamma_val * p / rho);
                double paraL = (gamma_val-1) / (2*c*c);

                double n_vector[2] = {0.0, 1.0};
                double l_vector[2] = {-1.0, 0.0};
                double qn = u*n_vector[0]+v*n_vector[1];
                double ql = u*l_vector[0]+v*l_vector[1];
                L_y[0][0] = 0.5*((gamma_val-1)*(u*u+v*v)/(2*c*c) + qn/c);
                L_y[0][1] = -0.5*((gamma_val-1)*u/(c*c) + n_vector[0]/c);
                L_y[0][2] = -0.5*((gamma_val-1)*v/(c*c) + n_vector[1]/c);
                L_y[0][3] = paraL;
                L_y[1][0] = 1-paraL*(u*u+v*v);
                L_y[1][1] = 2*u * paraL;
                L_y[1][2] = 2*v * paraL;
                L_y[1][3] = -2 * paraL;
                L_y[2][0] = 0.5*((gamma_val-1)*(u*u+v*v)/(2*c*c) - qn/c);
                L_y[2][1] = -0.5*((gamma_val-1)*u/(c*c) - n_vector[0]/c);
                L_y[2][2] = -0.5*((gamma_val-1)*v/(c*c) - n_vector[1]/c);
                L_y[2][3] = paraL;
                L_y[3][0] = -ql;
                L_y[3][1] = l_vector[0];
                L_y[3][2] = l_vector[1];
                L_y[3][3] = 0.0;
                
                R_y[0][0] = 1.0;
                R_y[0][1] = 1.0;
                R_y[0][2] = 1.0;
                R_y[0][3] = 0.0;
                R_y[1][0] = u - c*n_vector[0];
                R_y[1][1] = u;
                R_y[1][2] = u+c*n_vector[0];
                R_y[1][3] = l_vector[0];
                R_y[2][0] = v-c*n_vector[1];
                R_y[2][1] = v;
                R_y[2][2] = v+c*n_vector[1];
                R_y[2][3] = l_vector[1];
                R_y[3][0] = H - qn*c;
                R_y[3][1] = 0.5*(u*u + v*v);
                R_y[3][2] = H + qn*c;
                R_y[3][3] = ql;
                
                for(int pt = 0; pt < 6; pt++){
                    U_mat[0] = U_pt[0][pt];
                    U_mat[1] = U_pt[1][pt];
                    U_mat[2] = U_pt[2][pt];
                    U_mat[3] = U_pt[3][pt];
                    double *u_mat = matmul(L_y, U_mat);
                    u_pt[0][pt] = u_mat[0];
                    u_pt[1][pt] = u_mat[1];
                    u_pt[2][pt] = u_mat[2];
                    u_pt[3][pt] = u_mat[3];

                    F_mat[0] = F_pt[0][pt];
                    F_mat[1] = F_pt[1][pt];
                    F_mat[2] = F_pt[2][pt];
                    F_mat[3] = F_pt[3][pt];
                    double *f_mat = matmul(L_y, F_mat);
                    f_pt[0][pt] = f_mat[0];
                    f_pt[1][pt] = f_mat[1];
                    f_pt[2][pt] = f_mat[2];
                    f_pt[3][pt] = f_mat[3];
                    
                    delete[] u_mat;
                    delete[] f_mat;
                }
                
                // flux splitting
                for(int k = 0; k < 4; k++){
                    for(int pt = 0; pt < 5; pt++){
                        f_p[k][pt] = 0.5 * (f_pt[k][pt] + Lambda_y[k]*u_pt[k][pt]);
                        f_m[k][pt] = 0.5 * (f_pt[k][5-pt] - Lambda_y[k]*u_pt[k][5-pt]);
                    }
                }
                
                // TENO reconstruction
                for(int k = 0; k < 4; k++){
                    
                    double AAAA=Tianpy[k][i][j];
                    double BBBB=Tianmy[k][i][j];
                    
//                    f_hat[k] = CWENOZ_P(f_p[k][0], f_p[k][1], f_p[k][2], f_p[k][3], f_p[k][4]) + CWENOZ_P(f_m[k][0], f_m[k][1], f_m[k][2], f_m[k][3], f_m[k][4]);
                    
//                    f_hat[k] = TENO5(f_p[k][0], f_p[k][1], f_p[k][2], f_p[k][3], f_p[k][4]) + TENO5(f_m[k][0], f_m[k][1], f_m[k][2], f_m[k][3], f_m[k][4]);
                    
                    f_hat[k] = CENO_P(f_p[k][0], f_p[k][1], f_p[k][2], f_p[k][3], f_p[k][4], &AAAA, dy) + CENO_P(f_m[k][0], f_m[k][1], f_m[k][2], f_m[k][3], f_m[k][4], &BBBB, dy);
//                    f_hat[k] = MRWENO(f_p[k][0], f_p[k][1], f_p[k][2], f_p[k][3], f_p[k][4]) + MRWENO(f_m[k][0], f_m[k][1], f_m[k][2], f_m[k][3], f_m[k][4]);
                    
                    Tianpy[k][i][j] = AAAA;
                    Tianmy[k][i][j] = BBBB;
                }
                
                // projection back
                double *F_hat_mat = matmul(R_y, f_hat);
                G_hat[0][i][j] = F_hat_mat[0];
                G_hat[1][i][j] = F_hat_mat[1];
                G_hat[2][i][j] = F_hat_mat[2];
                G_hat[3][i][j] = F_hat_mat[3];
                delete[] F_hat_mat;
                
                for (int k = 0; k < 4; k++) {
                    delete[] U_pt[k];
                    delete[] F_pt[k];
                    delete[] L_y[k];
                    delete[] R_y[k];
                    delete[] u_pt[k];
                    delete[] f_pt[k];
                    delete[] f_p[k];
                    delete[] f_m[k];
                }
                delete[] U_pt;
                delete[] F_pt;
                delete[] L_y;
                delete[] R_y;
                delete[] u_pt;
                delete[] f_pt;
                delete[] f_p;
                delete[] f_m;
                delete[] f_hat;
                delete[] U_mat;
                delete[] F_mat;
                
            }
        }

 

   
        // compute LU
#pragma omp parallel for collapse(2)
        for(int k = 0; k < 4; k++){
            for(int i = 5; i < m+5; i++){
                for(int j = 5; j < n+5; j++){
                    LU[k][i][j] = - (F_hat[k][i][j] - F_hat[k][i-1][j]) / dx - (G_hat[k][i][j] - G_hat[k][i][j-1]) / dy;
                }
            }
        }
        
        // optimal 3rd order SSP Runge-Kutta
        switch (rk){
            case 0:
                for(int k = 0; k < 4; k++){
                    for(int i = 5; i < m+5; i++){
                        for(int j = 5; j < n+5; j++){
                            U[k][i][j] = U[k][i][j] + dt*LU[k][i][j];
                        }
                    }
                }
                break;
            case 1:
                for(int k = 0; k < 4; k++){
                    for(int i = 5; i < m+5; i++){
                        for(int j = 5; j < n+5; j++){
                            U[k][i][j] = 0.75 * U_1[k][i][j] + 0.25 * U[k][i][j] + 0.25 * dt*LU[k][i][j];
                        }
                    }
                }
                break;
            case 2:
                for(int k = 0; k < 4; k++){
                    for(int i = 5; i < m+5; i++){
                        for(int j = 5; j < n+5; j++){
                            U[k][i][j] = U_1[k][i][j] / 3.0 + U[k][i][j]* 2.0/3.0 + dt*LU[k][i][j] *2.0/3.0;
                        }
                    }
                }
                double Residual = 0.0;
                for(int k = 0; k < 4; k++){
                    for(int i = 5; i < m+5; i++){
                        for(int j = 5; j < n+5; j++){
                                Residual = Residual + fabs(U[k][i][j] - U_1[k][i][j])/dt;
                            }
                        }
                    }
                Residual = Residual/(4.0*m*n);
                cout <<run_t << " " <<log10(Residual)<< endl;
                ofstream outFile_res;
                outFile_res.open("res.txt", ios::app);
                outFile_res << setprecision(12) << run_t << " " <<log10(Residual) << endl;
                outFile_res.close();
                break;
        }
    }
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < m + 10; i++) {
            delete[] F[k][i];
            delete[] G[k][i];
            delete[] F_hat[k][i];
            delete[] G_hat[k][i];
            delete[] LU[k][i];
            delete[] U_1[k][i];
            delete[] dF_nu_dx[k][i];
            delete[] dG_nu_dy[k][i];
        }
        delete[] F[k];
        delete[] G[k];
        delete[] F_hat[k];
        delete[] G_hat[k];
        delete[] LU[k];
        delete[] U_1[k];
        delete[] dF_nu_dx[k];
        delete[] dG_nu_dy[k];
    }
    delete[] F;
    delete[] G;
    delete[] F_hat;
    delete[] G_hat;
    delete[] LU;
    delete[] U_1;
    delete[] dF_nu_dx;
    delete[] dG_nu_dy;
    
    return U;
}



// TENO5
double TENO5(double a, double b, double c, double d, double e){
    double C_T = 1.0e-5;
    double C = 1.0;
    double q = 6;
    double epsilon = 1.0e-12;
    
    double fhat_0 = (-b + 5*c + 2*d) / 6;
    double fhat_1 = (2*c + 5*d - e) / 6;
    double fhat_2 = (2*a - 7*b + 11*c) / 6;
    
    double beta_0 = (b-d)*(b-d) / 4 + (b-2*c+d)*(b-2*c+d) * 13/12;
    double beta_1 = (3*c-4*d+e)*(3*c-4*d+e) / 4 + (c-2*d+e)*(c-2*d+e) * 13/12;
    double beta_2 = (a-4*b+3*c)*(a-4*b+3*c) / 4 + (a-2*b+c)*(a-2*b+c) * 13/12;
    
    double tau_5 = fabs(beta_2 - beta_1);
    
    double gamma_val_0 = pow(C + tau_5/(beta_0+epsilon), q);
    double gamma_val_1 = pow(C + tau_5/(beta_1+epsilon), q);
    double gamma_val_2 = pow(C + tau_5/(beta_2+epsilon), q);
    
    double sum_gamma_val = gamma_val_0 + gamma_val_1 + gamma_val_2;
    double chi_0 = gamma_val_0 / sum_gamma_val;
    double chi_1 = gamma_val_1 / sum_gamma_val;
    double chi_2 = gamma_val_2 / sum_gamma_val;
    
    double delta_0 = 1, delta_1 = 1, delta_2 = 1;
    if(chi_0 < C_T){delta_0 = 0;}
    if(chi_1 < C_T){delta_1 = 0;}
    if(chi_2 < C_T){delta_2 = 0;}
    
    double d_0 = 0.6, d_1 = 0.3, d_2 = 0.1;
    double sum_d = d_0*delta_0 + d_1*delta_1 + d_2*delta_2;
    double w_0 = d_0*delta_0 / sum_d;
    double w_1 = d_1*delta_1 / sum_d;
    double w_2 = d_2*delta_2 / sum_d;
    
    return w_0*fhat_0 + w_1*fhat_1 + w_2*fhat_2;
}

double CWENOZ_P(double a, double b, double c, double d, double e)
{
    double p1= -(3.0*e-27.0*d-47.0*c+13.0*b-2.0*a)/60.0;
    double p2= (3.0*c-b)/2.0;
    double p3= (d+c)/2.0;

    double k1=0.98;
    double k2=0.01;
    double k3=0.01;

    p1=k1*(p1/k1-p2*k2/k1-p3*k3/k1)+k2*p2+k3*p3;


    double s11 = (1727.0*e*e)/1260.0-(51001.0*d*e)/5040.0+(7547.0*c*e)/560.0
    -(38947.0*b*e)/5040.0+(8209.0*a*e)/5040.0+(104963.0*d*d)/5040.0-(24923.0*c*d)/420.0
    +(89549.0*b*d)/2520.0-(38947.0*a*d)/5040.0+(77051.0*c*c)/1680.0-(24923.0*b*c)/420.0
    +(7547.0*a*c)/560.0+(104963.0*b*b)/5040.0-(51001.0*a*b)/5040.0+(1727.0*a*a)/1260.0;
    double s22 =c*c-2.0*b*c+b*b;
    double s33 =d*d-2.0*c*d+c*c;

    double tau = pow((fabs(s11-s22)+fabs(s11-s33))/2.0,2.0);

    double a1 = k1*(1.0+tau/(s11 + 1.0e-6));
    double a2 = k2*(1.0+tau/(s22 + 1.0e-6));
    double a3 = k3*(1.0+tau/(s33 + 1.0e-6));

    double w1 = a1 / (a1 + a2 + a3);
    double w2 = a2 / (a1 + a2 + a3);
    double w3 = a3 / (a1 + a2 + a3);

    return w1*(p1/k1-p2*k2/k1-p3*k3/k1)+w2*p2+w3*p3;
//    return c;
}



double CENO_P(double a, double b, double c, double d, double e, double *delta,double tre)
{
 int k;
 double b1, b2, b3,b4,a4;
 double a1, a2, a3, w1, w2, w3;
    double Variation1,Variation2,Variation3,Variation4;
 double bb1, bb2, bb3,ff1,ff2,rrr,rrr2,rrrfu1,HHCT;


   bb1=double(int(*delta/1000.0));
 bb2=double(int((*delta-bb1*1000.0)/100.));
 bb3=double(int((*delta-bb1*1000.0-bb2*100.0)/10.));

 double s11 = (1727.0*e*e)/1260.0-(51001.0*d*e)/5040.0+(7547.0*c*e)/560.0
 -(38947.0*b*e)/5040.0+(8209.0*a*e)/5040.0+(104963.0*d*d)/5040.0-(24923.0*c*d)/420.0
 +(89549.0*b*d)/2520.0-(38947.0*a*d)/5040.0+(77051.0*c*c)/1680.0-(24923.0*b*c)/420.0
 +(7547.0*a*c)/560.0+(104963.0*b*b)/5040.0-(51001.0*a*b)/5040.0+(1727.0*a*a)/1260.0;
 double s22 =c*c-2.0*b*c+b*b;
 double s33 =d*d-2.0*c*d+c*c;

    double s55 = fabs(s11-s22)+fabs(s11-s33);

 double hgh=tre*tre*tre;
    a1 = pow(1.0 + s55/(s11+hgh), 6.0);
    a2 = pow(1.0 + s55/(s22+hgh), 6.0);
    a3 = pow(1.0 + s55/(s33+hgh), 6.0);

 b1 = a1/(a1 + a2 + a3);
 b2 = a2/(a1 + a2 + a3);
 b3 = a3/(a1 + a2 + a3);

if (bb1>0.5)
{
 b1 = b1 < 1.0e-10 ? 0. : 1.;
}
else
{
 b1 = b1 < 1.0e-1 ? 0. : 1.;
}


if (bb2>0.5)
{
 b2 = b2 < 1.0e-8 ? 0. : 1.;
}
else
{
 b2 = b2 < 1.0e-1 ? 0. : 1.;
}

if (bb3>0.5)
{
 b3 = b3 < 1.0e-8 ? 0. : 1.;
}
else
{
 b3 = b3 < 1.0e-1 ? 0. : 1.;
}

 Variation1 = -(3.0*e-27.0*d-47.0*c+13.0*b-2.0*a)/60.0;
    Variation2 = (3.0*c-b)/2.0;
 Variation3 = (d+c)/2.0;

if (b1>0.5)
{
 b2=0.0;
 b3=0.0;
 ff1=Variation1;
}
else
{
 b1 = 0.0;
 a2 = b2/3.0;
 a3 = b3*2.0/3.0;
 w2=a2/(a2 + a3);
 w3=a3/(a2 + a3);
 ff1=w2*Variation2+ w3*Variation3;
}

 *delta= b1*1000.0+b2*100.0+b3*10.0+1.0;
 return ff1;
}


// WENO5
double WENO5(double a, double b, double c, double d, double e){
    double epsilon = 1.0e-12;
    
    double fhat_0 = (2*a - 7*b + 11*c) / 6;
    double fhat_1 = (-b + 5*c + 2*d) / 6;
    double fhat_2 = (2*c + 5*d - e) / 6;
    
    double beta_0 = (a-4*b+3*c)*(a-4*b+3*c) / 4 + (a-2*b+c)*(a-2*b+c) * 13/12;
    double beta_1 = (b-d)*(b-d) / 4 + (b-2*c+d)*(b-2*c+d) * 13/12;
    double beta_2 = (3*c-4*d+e)*(3*c-4*d+e) / 4 + (c-2*d+e)*(c-2*d+e) * 13/12;
    
    double w_0 = 1/((epsilon + beta_0)*(epsilon + beta_0));
    double w_1 = 6/((epsilon + beta_1)*(epsilon + beta_1));
    double w_2 = 3/((epsilon + beta_2)*(epsilon + beta_2));
    double sum_w = w_0 + w_1 + w_2;
    w_0 = w_0 / sum_w;
    w_1 = w_1 / sum_w;
    w_2 = w_2 / sum_w;
    
    return w_0*fhat_0 + w_1*fhat_1 + w_2*fhat_2;
}



double MRWENO(double a, double b, double c, double d, double e)
{
    double p5=-(3.0*e-27.0*d-47.0*c+13.0*b-2.0*a)/60.0;
    double p3= 1.0/6.0*(-b +5.0* c + 2.0*d);
    double p1=c;

    double ga12=1.0/(10.0+1.0);
    double ga22=10.0/(10.0+1.0);
    double ga13=1.0/(100.0+10.0+1.0);
    double ga23=10.0/(100.0+10.0+1.0);
    double ga33=100.0/(100.0+10.0+1.0);

    double q1=p1;
    double q2=p3/ga22-q1*ga12/ga22;
    double q3=p5/ga33-(ga13*q1+ga23*q2)/ga33;

    double beita3 = (1727.0*e*e)/1260.0-(51001.0*d*e)/5040.0+(7547.0*c*e)/560.0
    -(38947.0*b*e)/5040.0+(8209.0*a*e)/5040.0+(104963.0*d*d)/5040.0-(24923.0*c*d)/420.0
    +(89549.0*b*d)/2520.0-(38947.0*a*d)/5040.0+(77051.0*c*c)/1680.0-(24923.0*b*c)/420.0
    +(7547.0*a*c)/560.0+(104963.0*b*b)/5040.0-(51001.0*a*b)/5040.0+(1727.0*a*a)/1260.0;

    double beita2 = 13.0*(b - 2.0*c + d)*(b - 2.0*c + d)
       + 3.0*(b - d)*(b - d);

    double gama0,gama1,sita0,sita1,sita,beita1,tau;
    double ksi0=(c*c-2.0*b*c+b*b);
    double ksi1=(d*d-2.0*c*d+c*c);
    if (ksi0>=ksi1)
    {
        gama0 =1.0;
    }
    else
    {
        gama0 =10.0;
    }
    gama1=11.0-gama0;
    gama0=gama0/(gama0+gama1);
    gama1=gama1/(gama0+gama1);
    sita0=gama0*(1.0+pow(ksi0-ksi1,2.0)/(ksi0+1.0e-6));
    sita1=gama1*(1.0+pow(ksi0-ksi1,2.0)/(ksi1+1.0e-6));
    sita=sita0+sita1;
    beita1=pow(sita0*(c-b)+sita1*(d-c),2.0)/(sita*sita);
    tau= pow((abs(beita3-beita1)+abs(beita3-beita2))/2.0,2.0);

    double a13=ga13*(1.0+tau/(beita1+1.0e-6));
    double a23=ga23*(1.0+tau/(beita2+1.0e-6));
    double a33=ga33*(1.0+tau/(beita3+1.0e-6));

    double w13=a13/(a13+a23+a33);
    double w23=a23/(a13+a23+a33);
    double w33=a33/(a13+a23+a33);

    return w13*q1+w23*q2+w33*q3;
}



double norm_Linf(double** f) {
    double maxval = 0.0;
    for (int i = 5; i < m + 5; i++) {
        for (int j = 5; j < n + 5; j++) {
            maxval = std::max(maxval, std::fabs(f[i][j]));
        }
    }
    return maxval;
}


//first-order derivative
double diff_1(double a1, double a2, double a4, double a5, double h){
    return ( a1-8.0*a2+8.0*a4-a5 ) / (12.0*h);
}
// second-order derivative (complex)
double diff_2(double a1, double a2, double a3, double a4, double a5, double b1, double b2, double b3, double b4, double b5, double h){
    return ( (a1-25.0*a2-a3+a4)*b1 + (-27.0*a1+218.0*a2+220.0*a3-26.0*a4-a5)*b2 + (27.0*a1-168.0*a2-438.0*a3-168.0*a4+27.0*a5)*b3 + (-a1-26.0*a2+220.0*a3+218.0*a4-27.0*a5)*b4 + (a2-a3-25.0*a4+a5)*b5 ) / (288.0*h*h);
}


double* matmul(double** A, double* B) {
    double* C = new double[4];
    for (int i = 0; i < 4; ++i) {
        C[i] = A[i][0] * B[0] + A[i][1] * B[1] + A[i][2] * B[2] + A[i][3] * B[3];
    }
    return C;
}


double initialization_rho(double x, double y){ return 1.0; }
double initialization_u(double x, double y){ return 4.0; }
double initialization_v(double x, double y){ return 0.0; }
double initialization_p(double x, double y){ return 1.0/1.4; }






