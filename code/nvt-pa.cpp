/* 
Population Annealing in the NVT Ensemble
Jared Callaham

3/28/16 - (1.0) Working on adapting to binary mixture. Starting with 1.0 polydispersity ratio
4/4     - (1.1) Moved annealStep cumulative variable resets to allow more than 1 sweep
4/11    - (1.2) Random seed input from command line (for array jobs)
5/6     - (2.0) Can now randomly initialize at nonzero density
5/12    - (2.1) Fixed initial density determination and density calculation in annealing step
5/25    - (2.2) Changed count to keep track of proportion of collisions between particles of same size.
                Now saves snapshots at specified densities
6/2     - (2.3) Continue compression after FINAL_DENSITY until jamming. Save 1% of systems (chosen randomly) at FINAL_DENSITY
2/13/17 - (2.4) Load annealing schedule from file (sweeps and packing fractions). 
2/14    - (2.5) Fixed loading from file, now allow non-integer number of sweeps. Fixed step counting
2/15    - (2.6b) Allowing input population schedule
3/7     - (2.7) Version for large trial submission. Population fluctuations, scheduled sweeps, save low pressure configurations
4/5     - (3.0) Simplified and commented code. Removed saving low pressure configurations

*/
#include <stdlib.h>
#include <map>
#include <vector>
#include <cmath>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "prng.hpp"
#include <omp.h>
using namespace std;

// Trial parameters
const int N = 60;			            // Number of particles in system
double chainLength = 2*sqrt(N);          // To move sqrt(N) particles per step (this dynamically updates)
const double sweepLength = sqrt(N);	    // Number of ECMC steps per sweep assuming about sqrt(N) displacements per step
const int R_NOM = (int) 1e7;                // Nominal population size
const int R_MAX = (int) 1.01e7;             // Array size for population (R_NOM + 10*sqrt(R_NOM))
double RATIO = 1.4;                         // Polydispersity ratio. Set to 1 for single-size hard spheres

const int NUM_THREADS = 128;				// Number of OpenMP threads
int jobNumber, randSeed;                // Identifying code for array trials
sitmo::prng_engine rng[R_MAX];

// Information about the current state of the systems
double particles[R_MAX][N][3];
double VOL = 1;                         // Simulation volume Fixed by packing fraction
double L = 1;                           // Box side length
double Z = 1;                           // Dynamic equation of state (measured and averaged across population)
double Q;                               // Normalization factor for annealing steps

// Information about the population
int R = R_NOM;                          // Current population size (can fluctuate)
int family[R_MAX];                      // "Family" identifier for each system
int dir[R_MAX][3];

// Annealing schedule
vector<double> densities;   // Schedule of packing fractions 
vector<double> sweeps;      // Schedule of sweeps
int t = 0;                  // Location in annealing schedule (NOTE: NOT TIME AS MEASURED IN ECMC SWEEPS)

// Statistics to save
vector<double> eos;         // ECMC equation of state
vector<double> norms;       // "Q" normalization factors for each step
vector<double> rho_t;       // Mean square family size
vector<int> pops;           // Population size

// Special modulo operators to wrap around map, including overloads for int and double
int mod(int i, const int bound){ return (i + bound) % bound; }
double mod(double i, const double bound){ return fmod(i + bound, bound); }

// Generate uniform random number on (0, 1)
double rndm(int sysIndex){ return (double) rng[sysIndex]() / rng[sysIndex].max(); }

// Randomly chooses the next direction of motion (dir[0]) for an event chain.
void updateDir(int sysIndex){
	dir[sysIndex][0] = rng[sysIndex]() % 3;
	dir[sysIndex][1] = (dir[sysIndex][0] + 1) % 3;
	dir[sysIndex][2] = (dir[sysIndex][0] + 2) % 3;
}

// Return the diameter of this particle given its index
double D(int p){
    return 1 + (p%2) * (RATIO - 1);
}

void init(){        // Initialize the population at infinite volume
    // Load annealing schedule. Format should be (packing fraction)\t(number of sweeps)\n
    ifstream f;
    f.open("nvt-schedule.dat");
    double phi, s, pop;

    while (!f.eof()) {
        f >> phi >> s;      // Read from annealing schedule file
        densities.push_back( phi );  // Add to schedule vectors
        sweeps.push_back( s );
        f.ignore(256, '\n');
    }
    f.close();

    // Seed parallel random number generators
	for (int n=0; n<R_MAX; n++) rng[n] = sitmo::prng_engine(rand());

    // Initialize systems
    for(int n=0; n<R; n++){
	    family[n] = n;
        updateDir(n);    // Initialize directions

        // Generate N random points on (0, 1). This is automtically valid at phi=0
	    for (int p = 0; p<N; p++)
		    for (int i = 0; i < 3; i++)
                particles[n][p][i] = rndm(n);
    }

}

void replicate(int replace, int with){      // Copies a system (used during annealStep)
	for (int p = 0; p<N; p++){
		particles[replace][p][0] = particles[with][p][0];
		particles[replace][p][1] = particles[with][p][1];
		particles[replace][p][2] = particles[with][p][2];
	}
    family[replace] = family[with];
}


double collision(int p, int n, int sysIndex){	// Check for collision between p and n
	// Names are arbitrary - 'x' is direction of motion and 'y', 'z' are treated identically
	double dx, dy, dz, contact;
    contact = pow(0.5*(D(p) + D(n)), 2);  // Center-center distance at hypothetical contact

	dy = fabs(particles[sysIndex][p][dir[sysIndex][1]] - particles[sysIndex][n][dir[sysIndex][1]]);
	dy = fmin(dy, L - dy);

	dz = fabs(particles[sysIndex][p][dir[sysIndex][2]] - particles[sysIndex][n][dir[sysIndex][2]]);
	dz = fmin(dz, L - dz);

	dx = particles[sysIndex][n][dir[sysIndex][0]] - particles[sysIndex][p][dir[sysIndex][0]];

	if (dy*dy + dz*dz >= contact)   // Neighbor too far away in 'y' or 'z' direction
		return VOL;	    	        // This will never be chosen as the colliding event

	else if (dx < 0)    // Neighbor "behind" particle under regular boundary conditions.
		dx += L;		// Accounts for periodic boundary conditions

	// If you make it this far, it's a valid collision
	return dx - sqrt(contact - dy*dy - dz*dz);  // 'x' distance to collision
}


double ecmcStep(int sysIndex){
	/*
        Choose a particle and direction (+x, +y, +z) at random.
        Particle either moves a given distance or collides with another particle.
        If a collision occurs, the colliding particle continues in the same direction.

        Return "excess" displacement (x_f - x_i)/(chainLength). See Michel et al (2015)
    */
	double toGo, eventDist, neighborDist, excessDisp;
	int p, c;
	p = rng[sysIndex]() % N;					// Choose a particle to move
	updateDir(sysIndex);                        // Choose a direction at random
	
	excessDisp = chainLength;                    // Track for pressure calculation
	toGo = chainLength;	                        // Remaining distance in this event-chain step

	while (toGo > 0.0){
		eventDist = VOL;				// Default "long" distance
		c = N;                          // "c" is the "colliding" particle which will be lifted in the next move

		for (int n = 0; n < N; n++){
			if (n != p){
				neighborDist = collision(p, n, sysIndex);
				if (neighborDist < eventDist){   
					eventDist = neighborDist;   // This is the new closest collision
					c = n;
				}
			}
		}

		if (eventDist > toGo){  // If toGo is less than the nearest particle, only complete the chain length
			eventDist = toGo;
			c = N;              // (c==N) means the event chain is complete
		}

        // Simple check that all is working well
        if (eventDist < -1e-8) cout << "Attempted invalid negative move of " << eventDist << endl;

		eventDist = fmax(eventDist, 0);			// Floating point tolerance. Probably not necessary

		// Move particle and account for periodic boundary conditions
		particles[sysIndex][p][dir[sysIndex][0]] = mod(particles[sysIndex][p][dir[sysIndex][0]] + eventDist, L);

		// Increment excess chain displacement for dynamic pressure calculation
		if (c!=N){
			excessDisp += particles[sysIndex][c][dir[sysIndex][0]] - particles[sysIndex][p][dir[sysIndex][0]];
			if (particles[sysIndex][c][dir[sysIndex][0]] < particles[sysIndex][p][dir[sysIndex][0]])
				excessDisp += L;  // Account for boundary conditions
		}

		p = c;                  // Update particle index so next particle will be the collider
		toGo -= eventDist;      // Decrement remaining distance in this event chain
	}
	return excessDisp/chainLength;
}

// Returns the maximum "base" diameter of the system that would still allow a valid configuration
//  Equivalent to the inverse of the factor that the system can be rescaled by and remain valid
double maxCompression(int sysIndex){
	double dx, dy, dz;
	double free = L;  // Default "big" number

	for(int p = 0; p < N-1; p++){
		for(unsigned n = p+1; n<N; n++){
			dx = fabs(particles[sysIndex][p][0] - particles[sysIndex][n][0]);
			dx = fmin(dx, L - dx);

			dy = fabs(particles[sysIndex][p][1] - particles[sysIndex][n][1]);
			dy = fmin(dy, L - dy);

			dz = fabs(particles[sysIndex][p][2] - particles[sysIndex][n][2]);
			dz = fmin(dz, L - dz);

			free = fmin(2 * sqrt(dx*dx + dy*dy + dz*dz) / (D(p) + D(n)), free);
		}
	}
	return free;
}

// Compress system with population annealing
void annealStep(){
	int i, j;

    // Calculate scale factor for change in density given by annealing schedule
    double scale;
    if (t == 0) scale = pow( N*M_PI * (1 + pow(RATIO, 3)) / (12*densities[1]), 1./3);   // Use volume to scale for first step
    else scale = pow( densities[t]/densities[t+1], 1./3 );

    int weight[R_MAX] = {0}; // Statistical weight is 0 if configuration is valid at new density or 1 otherwise
    #pragma omp parallel for
    for (i = 0; i < R; i++)
        weight[i] = (maxCompression(i) > 1/scale);       // If system can be compressed this much, weight is 1

    // Sum weights to calculate normalization factor
    Q = 0;
    for (i = 0; i < R; i++)
        Q += weight[i];
    Q = Q/R;            // Normalize by current population size

    // Choose number of replicas for each system. See Wang, Machta, Katzgraber (2015)
    double tau = ( (double) R_NOM/R)/Q;     // Expected number of replicas (constant across valid configurations)
    int n[R_MAX];                           // Actual numbers of replicas for each system. 
    vector<int> empty, copy, keep;          // Vectors to help with replication
    int R_new = 0;                          // New population size
    for (i=0; i<R; i++){
        if (weight[i] == 0) {               // Invalid configuration at new density
            n[i] = 0;
            empty.push_back(i);             // No longer any replica at this index
        }
        else {                              // Valid configuration at new density
            n[i] = (int) tau;               // Floor of expectation, probability 1 - (tau - floor(tau))
            if ( rndm(i) < tau - n[i] ) n[i] += 1;      // Ceiling of expectation, probability tau - floor(tau)
            R_new += n[i];

            // Add to vectors to help with replication
            keep.push_back(i);                 // At least one replica of this index
            for(j=1; j<n[i]; j++) copy.push_back(i);        // More than one replica of this index
        }
    }
    for (i=R; i<R_MAX; i++) empty.push_back(i);  // i>R are empty

    // Make copies as necessary. "emtpy" vector should not run out of entries
    while ( !copy.empty() ) {
        replicate( empty[0], copy[0] );
        empty.erase( empty.begin() );
        copy.erase( copy.begin() );
    }

    // Fill in "holes" in population
    while (empty[0] < R_new) {
        replicate( empty[0], keep.back() );
        keep.pop_back();
        empty.erase(empty.begin());
    }

    // Now actually change the size of the box
    L = L*scale;
    VOL = pow(L, 3);
    R = R_new;
    t++;    // Move to new packing fraction in annealing schedule

	// Compress all systems by rescaling particle locations
    for (i = 0; i<R; i++){
        for (j = 0; j<N; j++){
            particles[i][j][0] = particles[i][j][0]*scale;
            particles[i][j][1] = particles[i][j][1]*scale;
            particles[i][j][2] = particles[i][j][2]*scale;
        }
    }

    // Actual Monte Carlo steps happen here.
    int numChains = sweepLength*sweeps[t];          // Number of event chains at this density
    double disp[R_MAX] = {0};                       // To save output while running in parallel
    #pragma omp parallel for
    for (i = 0; i < R; i++)
        for (int chain = 0; chain < numChains; chain++)
	        disp[i] += ecmcStep(i);

    // Calculate dynamic equation of state by averaging excess displacement across chains and population
    Z = 0;
    for (i = 0; i < R; i++)
        Z += disp[i];
    Z = Z / (R*numChains);

    chainLength = sqrt(N/2) / (Z - 1);    // Update chain length based on eos. See appendix to 2017 paper
}

// Calculate mean square family size rho_t
double familySize(){
	vector<int> familyTable = vector<int>(R_NOM, 0);
	for (int i = 0; i < R; i ++)
		familyTable[family[i]]++;
	double rho = 0;
	for (unsigned int i = 0; i < R_NOM; i++)
  	    rho += pow((double) familyTable[i]/R, 2);
	return R_NOM*rho;
}

// Store current observable vectors, print values, and save some of them
void save(){
    cout << t << "\t" << densities[t] << "\t" << Z << "\t" << R << "\t" << VOL << "\t" << endl;

    eos.push_back(Z);
    norms.push_back(Q);
    rho_t.push_back(familySize());
    pops.push_back(R);

    string fileName;
    // If no jobNumber is assigned, store output files in the source directory
    if (jobNumber == -1) fileName = string("Z.dat");
    // Otherwise, save in directory for density, and data files xx.dat, where xx is the job number
    else{
	    char jobChar[21];
	    sprintf(jobChar, "eos/%02d", jobNumber);
	    fileName = jobChar + string(".dat");
    }

    FILE * fout;
    fout = fopen(fileName.c_str(), "w");

    // First line contains trial parameters
    fprintf(fout, "N=%d \t R=%d \t Save [density, Z, Q, rho_t, R]\n", N, R_NOM);
    for (unsigned i = 0; i < eos.size(); i++)
	    fprintf(fout, "%.7f \t %.7f \t %.4f \t %.4f \t %d \n", densities[i], eos[i], norms[i], rho_t[i], pops[t]);

    fclose(fout);
}


int main (int argc, char* argv[]){
    // For array jobs, call with  ./a.out jobNumber $RANDOM  or similar
	if (argc < 2){
		jobNumber = -1;
        randSeed = time(NULL);
    }
	else{
		jobNumber = atoi(argv[1]);
        randSeed = atoi(argv[2]);              // To avoid duplicate random seeds
    }

	cout << "Configuring Systems" << endl;
    cout << "Random seed: " << randSeed << endl;
	srand(randSeed);					// rand() will only be used to seed the prng engines

	omp_set_num_threads(NUM_THREADS);
    init();                             // Intitializes population and seeds prng engines

	cout << "Annealing" << endl;
    while (t < densities.size()){
        annealStep();                   // annealStep() performs ECMC moves, resamples, and increases packing fraction
        save();                         // Save statistics in external data file
	}

	cout << "Completed annealing." << endl;
    return 0;
}
