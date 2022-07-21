/*

miniAD is a miniapp of the GPU version of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

AutoDock is a Trade Mark of the Scripps Research Institute.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/

#include "kernels.hpp"
#include "calcenergy.hpp"
#include "auxiliary_genetic.hpp"

// if defined, new (experimental) SW genotype moves that are dependent
// on nr of atoms and nr of torsions of ligand are used
#define SWAT3 // Third set of Solis-Wets hyperparameters by Andreas Tillack

void gpu_perform_LS( uint32_t pops_by_runs,
		     uint32_t work_pteam, 
 		     float* pMem_conformations_next, 
		     float* pMem_energies_next, 
	 	     GpuData& cData, 
		     GpuDockparameters dockpars )

//The GPU global function performs local search on the pre-defined entities of conformations_next.
//The number of blocks which should be started equals to num_of_lsentities*num_of_runs.
//This way the first num_of_lsentities entity of each population will be subjected to local search
//(and each block carries out the algorithm for one entity).
//Since the first entity is always the best one in the current population,
//it is always tested according to the ls probability, and if it not to be
//subjected to local search, the entity with ID num_of_lsentities is selected instead of the first one (with ID 0).
{

    #pragma omp target teams distribute
//     num_teams(pops_by_runs) thread_limit(work_pteam)
    for (int idx = 0; idx < pops_by_runs; idx++)	
    {  //for teams

         float genotype_candidate[ACTUAL_GENOTYPE_LENGTH];
         float genotype_deviate  [ACTUAL_GENOTYPE_LENGTH];
         float genotype_bias     [ACTUAL_GENOTYPE_LENGTH];
         float rho;
         int   cons_succ;
         int   cons_fail;
         int   iteration_cnt;
         int   evaluation_cnt;
         float3struct calc_coords[MAX_NUM_OF_ATOMS];
         float offspring_genotype[ACTUAL_GENOTYPE_LENGTH];
         float offspring_energy;
         int entity_id;
	 float candidate_energy;
	 float partial_energy[NUM_OF_THREADS_PER_BLOCK];
/*
	 #pragma omp allocate(genotype_candidate) allocator(omp_pteam_mem_alloc)
	 #pragma omp allocate(genotype_deviate) allocator(omp_pteam_mem_alloc)
	 #pragma omp allocate(genotype_bias) allocator(omp_pteam_mem_alloc)
	 #pragma omp allocate(rho) allocator(omp_pteam_mem_alloc)
	 #pragma omp allocate(cons_succ) allocator(omp_pteam_mem_alloc)
	 #pragma omp allocate(cons_fail) allocator(omp_pteam_mem_alloc)
	 #pragma omp allocate(iteration_cnt) allocator(omp_pteam_mem_alloc)
	 #pragma omp allocate(evaluation_cnt) allocator(omp_pteam_mem_alloc)
	 #pragma omp allocate(calc_coords) allocator(omp_pteam_mem_alloc)
	 #pragma omp allocate(offspring_genotype) allocator(omp_pteam_mem_alloc)
	 #pragma omp allocate(offspring_energy) allocator(omp_pteam_mem_alloc)
	 #pragma omp allocate(entity_id) allocator(omp_pteam_mem_alloc)
  */
         #pragma omp parallel for
         for(int j =0; j< work_pteam; j++ ){             
 
//	      float candidate_energy;
              int run_id;

	// Determining run ID and entity ID
	// Initializing offspring genotype
              run_id = idx / dockpars.num_of_lsentities;
	      if (j == 0)
	      {
                  entity_id = idx % dockpars.num_of_lsentities;

		// Since entity 0 is the best one due to elitism,
		// it should be subjected to random selection
		if (entity_id == 0) {
			// If entity 0 is not selected according to LS-rate,
			// choosing an other entity
			if (100.0f*gpu_randf(cData.pMem_prng_states, idx, j) > dockpars.lsearch_rate) {
				entity_id = dockpars.num_of_lsentities;					
			}
		}

		offspring_energy = pMem_energies_next[run_id*dockpars.pop_size+entity_id];
		rho = 1.0f;
		cons_succ = 0;
		cons_fail = 0;
		iteration_cnt = 0;
		evaluation_cnt = 0;        
	      }
//--- thread barrier
    size_t offset = (run_id * dockpars.pop_size + entity_id) * GENOTYPE_LENGTH_IN_GLOBMEM;
	for (uint32_t gene_counter = j;
	     gene_counter < dockpars.num_of_genes;
	     gene_counter+= work_pteam) {
        offspring_genotype[gene_counter] = pMem_conformations_next[offset + gene_counter];
		genotype_bias[gene_counter] = 0.0f;
	}
    
//--- thread barrier

#ifdef SWAT3
	float lig_scale = 1.0f/sqrt((float)dockpars.num_of_atoms);
	float gene_scale = 1.0f/sqrt((float)dockpars.num_of_genes);
#endif
	while ((iteration_cnt < dockpars.max_num_of_iters) && (rho > dockpars.rho_lower_bound))
	{
		// New random deviate
		for (uint32_t gene_counter = j;
		     gene_counter < dockpars.num_of_genes;
		     gene_counter+= work_pteam)
		{
#ifdef SWAT3
			genotype_deviate[gene_counter] = rho*(2*gpu_randf(cData.pMem_prng_states, idx, j)-1)*(gpu_randf(cData.pMem_prng_states, idx, j) < gene_scale);

			// Translation genes
			if (gene_counter < 3) {
				genotype_deviate[gene_counter] *= dockpars.base_dmov_mul_sqrt3;
			}
			// Orientation and torsion genes
			else {
				if (gene_counter < 6) {
					genotype_deviate[gene_counter] *= dockpars.base_dang_mul_sqrt3 * lig_scale;
				} else {
					genotype_deviate[gene_counter] *= dockpars.base_dang_mul_sqrt3 * gene_scale;
				}
			}
#else
			genotype_deviate[gene_counter] = rho*(2*gpu_randf(cData.pMem_prng_states, idx, j)-1)*(gpu_randf(cData.pMem_prng_states, idx, j)<0.3f);

			// Translation genes
			if (gene_counter < 3) {
				genotype_deviate[gene_counter] *= dockpars.base_dmov_mul_sqrt3;
			}
			// Orientation and torsion genes
			else {
				genotype_deviate[gene_counter] *= dockpars.base_dang_mul_sqrt3;
			}
#endif
		}

		// Generating new genotype candidate
		for (uint32_t gene_counter = j;
		     gene_counter < dockpars.num_of_genes;
		     gene_counter+= work_pteam) {
			   genotype_candidate[gene_counter] = offspring_genotype[gene_counter] + 
							      genotype_deviate[gene_counter]   + 
							      genotype_bias[gene_counter];
		}

		// Evaluating candidate
//--- thread barrier

		// ==================================================================
                partial_energy[j] =
		gpu_calc_energy(
                genotype_candidate,
                //candidate_energy,
                run_id,
                calc_coords,
		j,
	        work_pteam,
		cData,
                dockpars
				);
		// =================================================================
               
		if (j == 0) {
			float energy_idx = 0.0f;
                        for(int i =0; i < work_pteam; i++)
                            energy_idx += partial_energy[i];
                        candidate_energy = energy_idx;
			evaluation_cnt++;
		}
//--- thread barrier

		if (candidate_energy < offspring_energy)	// If candidate is better, success
		{
			for (uint32_t gene_counter = j;
			     gene_counter < dockpars.num_of_genes;
			     gene_counter+= work_pteam)
			{
				// Updating offspring_genotype
				offspring_genotype[gene_counter] = genotype_candidate[gene_counter];

				// Updating genotype_bias
				genotype_bias[gene_counter] = 0.6f*genotype_bias[gene_counter] + 0.4f*genotype_deviate[gene_counter];
			}

			// Work-item 0 will overwrite the shared variables
			// used in the previous if condition
//--- thread barrier

			if (j == 0)
			{
				offspring_energy = candidate_energy;
				cons_succ++;
				cons_fail = 0;
			}
		}
		else	// If candidate is worser, check the opposite direction
		{
			// Generating the other genotype candidate
			for (uint32_t gene_counter = j;
			     gene_counter < dockpars.num_of_genes;
			     gene_counter+= work_pteam) {
				   genotype_candidate[gene_counter] = offspring_genotype[gene_counter] - 
								      genotype_deviate[gene_counter] - 
								      genotype_bias[gene_counter];
			}

			// Evaluating candidate
//--- thread barrier
//#pragma omp barrier
			// =================================================================
                	partial_energy[j] =
			gpu_calc_energy(
                	genotype_candidate,
                	//candidate_energy,
                	run_id,
                	calc_coords,
	        	j,
                	work_pteam,
  			cData,
                        dockpars
            		);
			// =================================================================

			if (j == 0) {
				float energy_idx = 0.0f;
				for(int i =0; i < work_pteam; i++)
                                    energy_idx += partial_energy[i];
                                candidate_energy = energy_idx;
				evaluation_cnt++;

				#if defined (DEBUG_ENERGY_KERNEL)
				printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n", "-ENERGY-KERNEL3-", "GRIDS", "INTRA", partial_interE[0], partial_intraE[0]);
				#endif
			}
//--- thread barrier

			if (candidate_energy < offspring_energy) // If candidate is better, success
			{
				for (uint32_t gene_counter = j;
				     gene_counter < dockpars.num_of_genes;
			       	     gene_counter+= work_pteam)
				{
					// Updating offspring_genotype
					offspring_genotype[gene_counter] = genotype_candidate[gene_counter];

					// Updating genotype_bias
					genotype_bias[gene_counter] = 0.6f*genotype_bias[gene_counter] - 0.4f*genotype_deviate[gene_counter];
				}

				// Work-item 0 will overwrite the shared variables
				// used in the previous if condition
//--- thread barrier

				if (j == 0)
				{
					offspring_energy = candidate_energy;
					cons_succ++;
					cons_fail = 0;
				}
			}
			else	// Failure in both directions
			{
				for (uint32_t gene_counter = j;
				     gene_counter < dockpars.num_of_genes;
				     gene_counter+= work_pteam)
					   // Updating genotype_bias
					   genotype_bias[gene_counter] = 0.5f*genotype_bias[gene_counter];

				if (j == 0)
				{
					cons_succ = 0;
					cons_fail++;
				}
			}
		}

		// Changing rho if needed
		if (j == 0)
		{
			iteration_cnt++;

			if (cons_succ >= dockpars.cons_limit)
			{
				rho *= LS_EXP_FACTOR;
				cons_succ = 0;
			}
			else
				if (cons_fail >= dockpars.cons_limit)
				{
					rho *= LS_CONT_FACTOR;
					cons_fail = 0;
				}
		}
//--- thread barrier
	}

	// Updating eval counter and energy
	if (j == 0) {
		cData.pMem_evals_of_new_entities[run_id*dockpars.pop_size+entity_id] += evaluation_cnt;
		pMem_energies_next[run_id*dockpars.pop_size+entity_id] = offspring_energy;
	}

	// Mapping torsion angles and writing out results
        offset = (run_id*dockpars.pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM;
	for (uint32_t gene_counter = j;
	     gene_counter < dockpars.num_of_genes;
	     gene_counter+= work_pteam) {
        if (gene_counter >= 3) {
		    map_angle(offspring_genotype[gene_counter]);
		}
        pMem_conformations_next[offset + gene_counter] = offspring_genotype[gene_counter];
	}

     }
   }// End for a set of teams
}


