/*

miniMDock is a miniapp of the GPU version of AutoDock 4.2 running a Lamarckian Genetic Algorithm
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




template<class Device>
int prepare_const_fields(Liganddata&			myligand_reference,
				 Dockpars*			mypars,
				 float*				cpu_ref_ori_angles,
				 ConstantsW<Device>& c)

//The function fills the constant memory field of the GPU (ADM FPGA)
//defined above (erased from here) and used during GPU docking,
//based on the parameters which describe the ligand,
//the docking parameters and the reference orientation angles.
//Short description of the field is as follows:

//atom_charges_const: stores the ligand atom charges.
//		      Element i corresponds to atom with atom ID i in myligand_reference.
//atom_types_const: stores the ligand atom type IDs according to myligand_reference.
//		    Element i corresponds to atom with ID i in myligand_reference.
//intraE_contributors_const: each three contiguous items describe an intramolecular contributor.
//			 The first two elements store the atom ID of the contributors according to myligand_reference.
//			 The third element is 0, if no H-bond can occur between the two atoms, and 1, if it can.
//VWpars_AC_const: stores the A or C van der Waals parameters.
//                 The element i*MAX_NUM_OF_ATYPES+j and j*MAX_NUM_OF_ATYPES+i corresponds to A or C in case of
//		   H-bond for atoms with type ID i and j (according to myligand_reference).
//VWpars_BD_const: stores the B or D van der Waals parameters similar to VWpars_AC_const.
//dspars_S_const: stores the S desolvation parameters.
//		  The element i corresponds to the S parameter of atom with type ID i
//		  according to myligand_reference.
//rotlist_const: stores the data describing the rotations for conformation calculation.
//		 Each element describes one rotation, and the elements are in a proper order,
//               considering that NUM_OF_THREADS_PER_BLOCK rotations will be performed in
//		 parallel (that is, each block of contiguous NUM_OF_THREADS_PER_BLOCK pieces of elements describe rotations that can
//		 be performed simultaneously).
//		 One element is a 32 bit integer, with bit 0 in the LSB position.
//		 Bit 7-0 describe the atom ID of the atom to be rotated (according to myligand_reference).
//		 Bit 15-7 describe the rotatable bond ID of the bond around which the atom is to be rotated (if this is not a general rotation)
//				 (bond ID is according to myligand_reference).
//		 If bit 16 is 1, this is the first rotation of the atom.
//		 If bit 17 is 1, this is a general rotation (so rotbond ID has to be ignored).
//		 If bit 18 is 1, this is a "dummy" rotation, that is, no rotation can be performed in this cycle
//		 (considering the other rotations which are being carried out in this period).
//ref_coords_x_const: stores the x coordinates of the reference ligand atoms.
//		      Element i corresponds to the x coordinate of
//					  atom with atom ID i (according to myligand_reference).
//ref_coords_y_const: stores the y coordinates of the reference ligand atoms similarly to ref_coords_x_const.
//ref_coords_z_const: stores the z coordinates of the reference ligand atoms similarly to ref_coords_x_const.
//rotbonds_moving_vectors_const: stores the coordinates of rotatable bond moving vectors. Element i, i+1 and i+2 (where i%3=0)
//								 correspond to the moving vector coordinates x, y and z of rotbond ID i, respectively
//								 (according to myligand_reference).
//rotbonds_unit_vectors_const: stores the coordinates of rotatable bond unit vectors similarly to rotbonds_moving_vectors_const.
//ref_orientation_quats_const: stores the quaternions describing the reference orientations for each run. Element i, i+1, i+2
//							   and i+3 (where i%4=0) correspond to the quaternion coordinates q, x, y and z of reference
//							   orientation for run i, respectively.
{
	int i, j;
	int type_id1, type_id2;
	float* floatpoi;
	char* charpoi;
	float phi, theta, genrotangle;
	float atom_charges[MAX_NUM_OF_ATOMS];
	char  atom_types[MAX_NUM_OF_ATOMS];
	char  intraE_contributors[3*MAX_INTRAE_CONTRIBUTORS];
	float reqm [ATYPE_NUM];
        float reqm_hbond [ATYPE_NUM];
	unsigned int atom1_types_reqm [ATYPE_NUM];
        unsigned int atom2_types_reqm [ATYPE_NUM];
	float VWpars_AC[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
	float VWpars_BD[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
	float dspars_S[MAX_NUM_OF_ATYPES];
	float dspars_V[MAX_NUM_OF_ATYPES];
	int   rotlist[MAX_NUM_OF_ROTATIONS];
	float ref_coords_x[MAX_NUM_OF_ATOMS];
	float ref_coords_y[MAX_NUM_OF_ATOMS];
	float ref_coords_z[MAX_NUM_OF_ATOMS];
	float rotbonds_moving_vectors[3*MAX_NUM_OF_ROTBONDS];
	float rotbonds_unit_vectors[3*MAX_NUM_OF_ROTBONDS];
	float ref_orientation_quats[4*MAX_NUM_OF_RUNS];

	// Added for calculating torsion-related gradients.
	// Passing list of rotbond-atoms ids to the GPU.
	// Contains the same information as processligand.h/Liganddata->rotbonds		

	// Each row corresponds to one rotatable bond of the ligand.
	// The rotatable bond is described with the indexes of the
	// two atoms which are connected to each other by the bond.
	// The row index is equal to the index of the rotatable bond.
	int   rotbonds [2*MAX_NUM_OF_ROTBONDS];

	// Contains the same information as processligand.h/Liganddata->atom_rotbonds
	// "atom_rotbonds": array that contains the rotatable bonds - atoms assignment.
	// If the element atom_rotbonds[atom index][rotatable bond index] is equal to 1,
	// it means,that the atom must be rotated if the bond rotates. A 0 means the opposite.

	// "rotbonds_atoms"
	int  rotbonds_atoms [MAX_NUM_OF_ATOMS*MAX_NUM_OF_ROTBONDS];

	// Each entry corresponds to a rotbond_id
	// The value of an entry indicates the number of atoms that rotate 
	// along with that rotbond_id
	int  num_rotating_atoms_per_rotbond [MAX_NUM_OF_ROTBONDS];
	// ------------------------------

	//charges and type id-s
	floatpoi = atom_charges;
	charpoi = atom_types;

	for (i=0; i < myligand_reference.num_of_atoms; i++)
	{
		*floatpoi = (float) myligand_reference.atom_idxyzq[i][4];
		*charpoi = (char) myligand_reference.atom_idxyzq[i][0];
		floatpoi++;
		charpoi++;
	}

	//intramolecular energy contributors
	myligand_reference.num_of_intraE_contributors = 0;
	for (i=0; i<myligand_reference.num_of_atoms-1; i++)
		for (j=i+1; j<myligand_reference.num_of_atoms; j++)
		{
			if (myligand_reference.intraE_contributors[i][j])
				myligand_reference.num_of_intraE_contributors++;
		}

	if (myligand_reference.num_of_intraE_contributors > MAX_INTRAE_CONTRIBUTORS)
	{
		printf("Error: number of intramolecular energy contributor is too high!\n");
		fflush(stdout);
		return 1;
	}

	charpoi = intraE_contributors;
	for (i=0; i<myligand_reference.num_of_atoms-1; i++)
		for (j=i+1; j<myligand_reference.num_of_atoms; j++)
		{
			if (myligand_reference.intraE_contributors[i][j] == 1)
			{
				*charpoi = (char) i;
				charpoi++;
				*charpoi = (char) j;
				charpoi++;
				type_id1 = (int) myligand_reference.atom_idxyzq [i][0];
				type_id2 = (int) myligand_reference.atom_idxyzq [j][0];

				if (is_H_bond(myligand_reference.atom_types[type_id1], myligand_reference.atom_types[type_id2]) != 0)
					*charpoi = (char) 1;
				else
					*charpoi = (char) 0;
				charpoi++;
			}
		}

        // Smoothed pairwise potentials
	// reqm, reqm_hbond: equilibrium internuclear separation for vdW and hbond
	for (i= 0; i<ATYPE_NUM/*myligand_reference.num_of_atypes*/; i++) {
		reqm[i]       = myligand_reference.reqm[i];
		reqm_hbond[i] = myligand_reference.reqm_hbond[i];

		atom1_types_reqm [i] = myligand_reference.atom1_types_reqm[i];
		atom2_types_reqm [i] = myligand_reference.atom2_types_reqm[i];
	}

	//van der Waals parameters
	for (i=0; i<myligand_reference.num_of_atypes; i++)
		for (j=0; j<myligand_reference.num_of_atypes; j++)
		{
			if (is_H_bond(myligand_reference.atom_types[i], myligand_reference.atom_types[j]) != 0)
			{
				floatpoi = VWpars_AC + i*myligand_reference.num_of_atypes + j;
				*floatpoi = (float) myligand_reference.VWpars_C[i][j];
				floatpoi = VWpars_AC + j*myligand_reference.num_of_atypes + i;
				*floatpoi = (float) myligand_reference.VWpars_C[j][i];

				floatpoi = VWpars_BD + i*myligand_reference.num_of_atypes + j;
				*floatpoi = (float) myligand_reference.VWpars_D[i][j];
				floatpoi = VWpars_BD + j*myligand_reference.num_of_atypes + i;
				*floatpoi = (float) myligand_reference.VWpars_D[j][i];
			}
			else
			{
				floatpoi = VWpars_AC + i*myligand_reference.num_of_atypes + j;
				*floatpoi = (float) myligand_reference.VWpars_A[i][j];
				floatpoi = VWpars_AC + j*myligand_reference.num_of_atypes + i;
				*floatpoi = (float) myligand_reference.VWpars_A[j][i];

				floatpoi = VWpars_BD + i*myligand_reference.num_of_atypes + j;
				*floatpoi = (float) myligand_reference.VWpars_B[i][j];
				floatpoi = VWpars_BD + j*myligand_reference.num_of_atypes + i;
				*floatpoi = (float) myligand_reference.VWpars_B[j][i];
			}
		}

	//desolvation parameters
	for (i=0; i<myligand_reference.num_of_atypes; i++)
	{
		dspars_S[i] = myligand_reference.solpar[i];
		dspars_V[i] = myligand_reference.volume[i];
	}

	//generate rotation list
	if (gen_rotlist(myligand_reference, rotlist) != 0)
	{
		printf("Error: number of required rotations is too high!\n");
		return 1;
	}

	//coordinates of reference ligand
	for (i=0; i < myligand_reference.num_of_atoms; i++)
	{
		ref_coords_x[i] = myligand_reference.atom_idxyzq[i][1];
		ref_coords_y[i] = myligand_reference.atom_idxyzq[i][2];
		ref_coords_z[i] = myligand_reference.atom_idxyzq[i][3];
	}

	//rotatable bond vectors
	for (i=0; i < myligand_reference.num_of_rotbonds; i++)
		for (j=0; j<3; j++)
		{
			rotbonds_moving_vectors[3*i+j] = myligand_reference.rotbonds_moving_vectors[i][j];
			rotbonds_unit_vectors[3*i+j] = myligand_reference.rotbonds_unit_vectors[i][j];
		}


	//reference orientation quaternions
	for (i=0; i<mypars->num_of_runs; i++)
	{
		//printf("Pregenerated angles for run %d: %f %f %f\n", i, cpu_ref_ori_angles[3*i], cpu_ref_ori_angles[3*i+1], cpu_ref_ori_angles[3*i+2]);

		phi = cpu_ref_ori_angles[3*i]*DEG_TO_RAD;
		theta = cpu_ref_ori_angles[3*i+1]*DEG_TO_RAD;
		genrotangle = cpu_ref_ori_angles[3*i+2]*DEG_TO_RAD;

		ref_orientation_quats[4*i+0] = sinf(genrotangle/2.0f)*sinf(theta)*cosf(phi);		//x
		ref_orientation_quats[4*i+1] = sinf(genrotangle/2.0f)*sinf(theta)*sinf(phi);		//y
		ref_orientation_quats[4*i+2] = sinf(genrotangle/2.0f)*cosf(theta);			//z
		ref_orientation_quats[4*i+3] = cosf(genrotangle/2.0f);					//q
/*
		// Shoemake genes
		// autodockdev/motions.py

		float u1, u2, u3;
		u1 = cpu_ref_ori_angles[3*i];
		u2 = cpu_ref_ori_angles[3*i+1];
		u3 = cpu_ref_ori_angles[3*i+2];

		ref_orientation_quats[4*i]   = sqrt(1-u1) * sinf(2*PI*u2);	//q
		ref_orientation_quats[4*i+1] = sqrt(1-u1) * cosf(2*PI*u2);	//x
		ref_orientation_quats[4*i+2] = sqrt(u1)   * sinf(2*PI*u3);	//y
		ref_orientation_quats[4*i+3] = sqrt(u1)   * cosf(2*PI*u3);	//z
*/
		//printf("Precalculated quaternion for run %d: %f %f %f %f\n", i, ref_orientation_quats[4*i], ref_orientation_quats[4*i+1], ref_orientation_quats[4*i+2], ref_orientation_quats[4*i+3]);
	}

	// Added for calculating torsion-related gradients.
	// Passing list of rotbond-atoms ids to the GPU.
	// Contains the same information as processligand.h/Liganddata->rotbonds
	for (i=0; i < myligand_reference.num_of_rotbonds; i++)
	{
		rotbonds [2*i]   = myligand_reference.rotbonds[i][0]; // id of first-atom
		rotbonds [2*i+1] = myligand_reference.rotbonds[i][1]; // id of second atom
	}

	// Contains the same information as processligand.h/Liganddata->atom_rotbonds
	// "atom_rotbonds": array that contains the rotatable bonds - atoms assignment.
	// If the element atom_rotbonds[atom index][rotatable bond index] is equal to 1,
	// it means,that the atom must be rotated if the bond rotates. A 0 means the opposite.
	for (i=0; i<MAX_NUM_OF_ROTBONDS; i++)
	{
		num_rotating_atoms_per_rotbond [i] = 0;
	}


	int* intpoi;
	//intpoi = rotbonds_atoms;

	for (i=0; i < myligand_reference.num_of_rotbonds; i++)
	{	
		// Pointing to the mem area corresponding to a given rotbond
		intpoi = rotbonds_atoms + MAX_NUM_OF_ATOMS*i;

		for (j=0; j < myligand_reference.num_of_atoms; j++)
		{
			// If an atom rotates with a rotbond, then
			// add its atom-id to the entry corresponding to the rotbond-id.
			// Also, count the number of atoms that rotate with a certain rotbond
			if (myligand_reference.atom_rotbonds [j][i] == 1){
				*intpoi = j;
				intpoi++;
				num_rotating_atoms_per_rotbond [i] ++;	
			}	

		}
	}

	int m;

	for (m=0;m<MAX_NUM_OF_ATOMS;m++){ c.interintra.atom_charges_const[m] = atom_charges[m]; }
	for (m=0;m<MAX_NUM_OF_ATOMS;m++){ c.interintra.atom_types_const[m]   = atom_types[m];   }

	for (m=0;m<3*MAX_INTRAE_CONTRIBUTORS;m++){ c.intracontrib.intraE_contributors_const[m] = intraE_contributors[m]; }

	for (m=0;m<ATYPE_NUM;m++){
		c.intra.reqm_const[m]	    = 0.5*reqm[m];
		c.intra.reqm_const[m+ATYPE_NUM]	    = reqm_hbond[m];
	}
	for (m=0;m<ATYPE_NUM;m++)				{ c.intra.atom1_types_reqm_const[m] = atom1_types_reqm[m]; }
	for (m=0;m<ATYPE_NUM;m++)				{ c.intra.atom2_types_reqm_const[m] = atom2_types_reqm[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES;m++)	{ c.intra.VWpars_AC_const[m]        = VWpars_AC[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES;m++)	{ c.intra.VWpars_BD_const[m]        = VWpars_BD[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES;m++)			{ c.intra.dspars_S_const[m]         = dspars_S[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES;m++)			{ c.intra.dspars_V_const[m]         = dspars_V[m]; }

	for (m=0;m<MAX_NUM_OF_ROTATIONS;m++) {
		c.rotlist.rotlist_const[m]  = rotlist[m];
/*		if(m!=0 && m%myligand_reference.num_of_atoms==0)
			printf("***\n");
		if(m!=0 && m%NUM_OF_THREADS_PER_BLOCK==0)
			printf("===\n");
		printf("%i (%i): %i -> atom_id: %i, dummy: %i, first: %i, genrot: %i, rotbond_id: %i\n",m,m%NUM_OF_THREADS_PER_BLOCK,rotlist[m],rotlist[m] & RLIST_ATOMID_MASK, rotlist[m] & RLIST_DUMMY_MASK,rotlist[m] & RLIST_FIRSTROT_MASK,rotlist[m] & RLIST_GENROT_MASK,(rotlist[m] & RLIST_RBONDID_MASK) >> RLIST_RBONDID_SHIFT);*/
	}

	for (m=0;m<MAX_NUM_OF_ATOMS;m++) {
		c.conform.ref_coords_const[3*m]		 = ref_coords_x[m];
		c.conform.ref_coords_const[3*m+1]	 = ref_coords_y[m];
		c.conform.ref_coords_const[3*m+2]	 = ref_coords_z[m];
	}
	for (m=0;m<3*MAX_NUM_OF_ROTBONDS;m++){ c.conform.rotbonds_moving_vectors_const[m]= rotbonds_moving_vectors[m]; }
	for (m=0;m<3*MAX_NUM_OF_ROTBONDS;m++){ c.conform.rotbonds_unit_vectors_const[m]  = rotbonds_unit_vectors[m]; }
	for (m=0;m<4*MAX_NUM_OF_RUNS;m++)    { c.conform.ref_orientation_quats_const[m]  = ref_orientation_quats[m]; }

	// Added for calculating torsion-related gradients.
	// Passing list of rotbond-atoms ids to the GPU.
	// Contains the same information as processligand.h/Liganddata->rotbonds
	for (m=0;m<2*MAX_NUM_OF_ROTBONDS;m++)			{ c.grads.rotbonds[m]			  = rotbonds[m]; }
	for (m=0;m<MAX_NUM_OF_ATOMS*MAX_NUM_OF_ROTBONDS;m++)	{ c.grads.rotbonds_atoms[m]                 = rotbonds_atoms[m]; }
	for (m=0;m<MAX_NUM_OF_ROTBONDS;m++)			{ c.grads.num_rotating_atoms_per_rotbond[m] = num_rotating_atoms_per_rotbond[m]; }
	return 0;
}

template<class Device>
void prepare_axis_correction( float* angle, float* dependence_on_theta, float* dependence_on_rotangle,
                                 AxisCorrectionW<Device>& axis_correction)
{
	for (int m=0;m<NUM_AXIS_CORRECTION;m++)
	{
		axis_correction.angle(m) = angle[m];
		axis_correction.dependence_on_theta(m) = dependence_on_theta[m];
		axis_correction.dependence_on_rotangle(m) = dependence_on_rotangle[m];
	}
}
