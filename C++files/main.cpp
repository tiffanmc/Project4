#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include "time.h"
#include <armadillo>
#include <limits>
#include <mpi.h>
#include <string>
using namespace std;
using namespace arma;

// DEFINING A FUNCTION TO COMPUTE THE ENERGY OF THE MATRIX
double computeEnergy(mat &Matrix, int N ) {
    double Energy= 0;
    for ( int i = 0; i < N; i++ )   //Checks if you're at one of the 4 corners of the matrix
    {
        for ( int j = 0; j < N; j++ )
        {
            int iNext = i+1;
            if (iNext > N-1) {
                iNext = 0;
            }
            int jNext = j+1;
            if (jNext > N-1) {
                jNext = 0;
            }
            int iPrev = i-1;
            if (iPrev < 0) {
                iPrev = N-1;
            }
            int jPrev = j-1;
            if (jPrev < 0) {
                jPrev = N-1;
            }
            Energy += Matrix(i,j)*(Matrix(iNext,j)+Matrix(i,jNext)+Matrix(iPrev,j)+Matrix(i,jPrev));
        }
    }
    return -Energy / 2.0;
}
// DEFINING A FUNCTION TO COMPUTE THE MAGNETIC MOMENT OF THE MATRIX
double computeMagMoment(mat &Matrix, int N ) {
    double MagMoment= 0;
    for ( int i = 0; i < N; i++ )
    {
        for ( int j = 0; j < N; j++ )
        {
            MagMoment += Matrix(i,j);
        }
    }
    return MagMoment;
}
// DEFINING A FUNCTION TO COMPUTE THE NEW ENERGY OF THE MATRIX ONLY BY CONSIDERING NEIGHBOURS TO THE FLIPPED SPIN
inline int periodic(int i, int limit, int add) {
    return (i+limit+add) % (limit);
}

// MAIN PROGRAM
int main(int argc, char *argv[])
{
    // MPI magic commands
    MPI_Init(NULL, NULL);   // Initialize the MPI environment
    int numberOfProcesses;  // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    int processID;          // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &processID);
    //cout << "Hello, I am processor " << processID << " of " << numberOfProcesses << " processes." << endl;

    int N = 140;            // Defining lattice size, lattice size is NxN
    int numTimeSteps = 1e6; // Defining number of timesteps within each Monte Carlo cycle

    //Defining temperatures we want to loop over to investigate phase transitions
    vector<double> temperatures;
    double T0 = 2.0; //Initial temperature
    double T1 = 2.4; //Final temperature
    int Ntemps = 32; //should be multiple of number of cores
    double deltaTemperature = (T1-T0) / Ntemps;

    MPI_Bcast(&Ntemps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&T0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&T1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&deltaTemperature, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Inserting a clock to measure the time of the algorithm
    clock_t start, end;
    start = clock();        // Starting the clock

    int numberOfTemperaturesPerProcessor = Ntemps / numberOfProcesses;
    int temperatureIndexStart = processID * numberOfTemperaturesPerProcessor;

    // Making a file of temperatures and expectation values used for plotting
    ofstream TempFile;
    std::string fileName = "ALLVARS_parallel";
    //making strings of IDs in order to have different txt files for each processor:
    fileName = fileName + std::to_string(processID) + ".txt";
    TempFile.open(fileName.c_str());
    cout << fileName.c_str() << endl;
    TempFile << setiosflags(ios::showpoint | ios::uppercase);

    srand(clock());     //Making numbers truly random, so the same ones are not repeated in each run
    srand (time(NULL)); //resetting random number
    int numberOfAcceptedSteps = 0;
    //double Temp = 1.0;    //used before we had temperature loop
    double k = 1.0;

    //Making initial random Matrix of -+1
    mat Matrix = zeros<mat>(N,N);
    for ( int i = 0; i < N; i++ ){
        for ( int j = 0; j < N; j++ )
        {
            int random_number = rand() % 2;
            if (random_number == 1) {
                Matrix(i,j) = 1;
            } else {
                Matrix(i,j) = -1;
            }
        }
    }
    //Making initial ordered Matrix of only +1
    mat MatrixOrdered = zeros<mat>(N,N);
    MatrixOrdered.fill(1);               // ordered matrix= all spin up
    //Matrix = MatrixOrdered; //Remove when random matrix!!

    // LOOPING OVER TEMPERATURES
    for(int i=0; i<numberOfTemperaturesPerProcessor; i++) {
        int index = temperatureIndexStart + i;
        double Temp = temperatures[index];
        //cout << "I am " << processID << " and will compute T=" << Temp << endl;

        // Morten's cool stuff to make our program faster
        double w_vec[17];
        for( int de =-8; de <= 8; de++) w_vec[de+8] = 0;
        for( int de =-8; de <= 8; de+=4) w_vec[de+8] = exp(-de/Temp);

        // Skipping the first timesteps to make sure we start taking data when TE is reached
        int skipTimesteps = 1000;

        //Setting initial values for parameters that are changed in loop
        double currentEnergy = computeEnergy(Matrix, N);
        double currentMagneticMoment= computeMagMoment(Matrix, N);
        double energySum = currentEnergy;
        double MagSum = currentMagneticMoment;
        double energySquaredSum = currentEnergy*currentEnergy;
        double MagSumSquared= currentMagneticMoment*currentMagneticMoment;

        // GOING THROUGH MATRIX AND FLIPPING ONE SPIN, TO TRY AND GET TO A LOWER STATE
        for(int t = 0; t < numTimeSteps; t++) {   //Monte Carlo loop, t is number of MC cycles
            bool sampleStuff = t > skipTimesteps; //Bolean array in order to skip steps before TE is reached
            for(int m = 0; m < N*N; m++){   //Extra loop to have more than one flip in each cycle
                int i = rand() % N;  //Going to a random position in the Matrix
                int j = rand() % N;
                //Flipping the spin at this position
                int deltaE =  2*Matrix(j,i)*
                        (Matrix(j,periodic(i,N,-1))+
                         Matrix(periodic(j,N,-1),i) +
                         Matrix(j,periodic(i,N,1)) +
                         Matrix(periodic(j,N,1),i));

                // METROPOLIS TEST
                // test if we have moved to a lower state
                double r = rand() / ((double) std::numeric_limits<int>::max());
                if ( r <= w_vec[deltaE+8] ) {
                    Matrix(j,i) *= -1;  // flip one spin and accept new spin config
                    currentMagneticMoment += (double) 2*Matrix(j,i);
                    currentEnergy += (double) deltaE;
                }
                /*OLD Metropolis
                if (deltaE <= 0) {           //accept step if delE<0 ie. we have moved to a lower state
                    numberOfAcceptedSteps++;
                    currentEnergy = Enew;
                    currentMagneticMoment = Magnew;
                } else {
                    double w = exp(-deltaE/Temp);
                    double r = rand() / ((double) std::numeric_limits<int>::max());
                    if (r <= w) {
                        numberOfAcceptedSteps++;    // also accept step if r<w
                        currentEnergy = Enew;
                        currentMagneticMoment = Magnew;
                    }else{
                        Matrix(i,j) *= -1;          // if not accepted, flip the spin we flipped back
                    }
                }
                */
            }
            /*Making an array of Energies in order to make a histogram of energies
            int NN = N*N*4;
            int histN = 0;
            double array_start = N*N*2;
            double array_end = 2.3;
            double array_hist[NN];
            int index = (array_start-currentEnergy)/8;
            if (sampleStuff && index > 0 && index < NN+1) {
                array_hist[index] ++;
                histN++;
            }*/
            //Making sums of quantities in order to compute expectation values, if TE is reached
            if(sampleStuff) {
                energySum += currentEnergy;
                MagSum += fabs(currentMagneticMoment);
                MagSumSquared += currentMagneticMoment*currentMagneticMoment;
                energySquaredSum += currentEnergy*currentEnergy;
            }
            //Old commands to write to files for previous plots
            //outputFile << currentEnergy << "  " << currentMagneticMoment << "  " << t << endl;
            //outputFile << numberOfAcceptedSteps << "  " << t << endl;
        }
        //Calculating the quantities we are interested in
        numTimeSteps -= skipTimesteps;
        double EE = energySum / ((double) numTimeSteps+1);      // Expectation energy
        double EE2= energySquaredSum/((double) numTimeSteps+1); // Expectation of energy squared
        double MM2= MagSumSquared /((double) numTimeSteps+1);   // Expectation of magnetic moment squared
        double MM= MagSum /(((double) numTimeSteps+1)) ;        // Expectation of magnetic moment
        double SpecificHeat= (EE2-(EE*EE))/ (k*Temp*Temp);      // Specific heat Cv
        double Suscept =  (MM2-(MM*MM))/ (k*Temp);              // Magnetic susceptibility

        //Writing array of Energies to file for histogram
        //int NN = N*N*4;
        //ofstream histFile;
        //char filename[10000];
        //sprintf(filename, "histT%.2f.dat", Temp);
        //histFile.open(filename);
        //histFile << setiosflags(ios::showpoint | ios::uppercase);
        //for (int i=0; i < NN; i++){
        //histFile << array_hist[i] / ((double) histN) <<endl;
        //}
        //histFile.close();

        /*Printing values to see results
        cout << "<E>   = " << EE << endl;
        cout << "<E^2> = " << EE2 << endl;
        cout << "C_v   = " << SpecificHeat << endl;
        cout << "<M>   = " << MM << endl;
        cout << "<M^2> = " << MM2 << endl;
        cout << "X     = " << Suscept << endl;
        cout << "variance = " << EE2-((EE)*(EE)) << endl << endl << endl;
        */
        //Writing to file for final plots of phase transitions
        TempFile << Temp << "  " << EE << "  " << MM << "  " << SpecificHeat << "  " << Suscept << endl;
    }
    end = clock();      // Ending the the clock
    TempFile.close();   // Closing file

    MPI_Finalize();     // Final MPI magic command
    return 0;
}
